# visualise_one_scene.py
"""
Colour the 3-D objects that BigGNN (or the DBSCAN overlap matcher) links to
each ScanScribe / Human caption for ONE scene.

SPACE/ENTER → next caption
q / ESC       → quit
"""

import json, argparse, torch, numpy as np, open3d as o3d, sys
from pathlib import Path
sys.path += ["../data_processing", "../../../"]
from scene_graph import SceneGraph
from data_distribution_analysis.helper import get_matching_subgraph
try:
    from model_graph2graph import BigGNN
except ImportError:
    BigGNN = None


def load_scene(scan_dir: Path):
    """
    Loads labels.instances.annotated.v2.ply and builds obj2faces.
    Returns a CPU TriangleMesh with full triangle_colors support.
    """
    ply_path = scan_dir / "labels.instances.annotated.v2.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing {ply_path}")
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    mesh.compute_vertex_normals()

    # ── FORCE CPU MESH ────────────────────────────────────────────────────────
    # If this is a CUDA triangle mesh, .cpu() will give you the CPU equivalent.
    try:
        mesh = mesh.cpu()
    except AttributeError:
        # already CPU; nothing to do
        pass

    # 2) pack per-vertex RGB→hex
    vc   = np.asarray(mesh.vertex_colors)
    vc8  = (vc * 255 + 0.5).astype(np.uint32)
    vhex = (vc8[:,0] << 16) | (vc8[:,1] << 8) | vc8[:,2]

    # 3) load objects.json from parent folder
    obj_json = scan_dir.parent / "objects.json"
    if not obj_json.exists():
        raise FileNotFoundError(f"{obj_json} not found")
    scans = {s["scan"]: s for s in json.load(open(obj_json))["scans"]}
    if scan_dir.name not in scans:
        raise KeyError(f"{scan_dir.name} missing in {obj_json}")

    color2oid = {
        int(o["ply_color"].lstrip("#"), 16): int(o["id"])
        for o in scans[scan_dir.name]["objects"]
    }

    # 4) vertex → object ID
    v_oid = np.array([color2oid.get(int(h), 0) for h in vhex], dtype=np.int64)

    # 5) triangle majority vote
    faces = np.asarray(mesh.triangles)
    f_oid = np.array([np.bincount(v_oid[f]).argmax() for f in faces],
                     dtype=np.int64)

    # 6) invert ⇒ obj2faces
    obj2faces = {}
    for fid, oid in enumerate(f_oid):
        if oid == 0: continue
        obj2faces.setdefault(int(oid), []).append(fid)
    obj2faces = {k: np.array(v, dtype=np.int64) for k,v in obj2faces.items()}
    if not obj2faces:
        raise RuntimeError("No labeled faces in " + str(scan_dir))

    return mesh, obj2faces


def colour_objects(mesh, obj2faces, matched, base=(0.6,0.6,0.6)):
    n_v   = len(mesh.vertices)
    vcols = np.tile(base, (n_v, 1))               # all vertices grey
    rng   = np.random.default_rng(42)

    for oid in matched:
        if oid not in obj2faces:
            continue
        tri_ids = obj2faces[oid]
        col     = rng.random(3)
        for tid in tri_ids:                       # set all 3 vertices of face
            for vid in mesh.triangles[tid]:
                vcols[int(vid)] = col

    mesh.vertex_colors = o3d.utility.Vector3dVector(vcols)
    return mesh


def to_legacy_cpu(mesh_t):
    """Convert any Tensor/CUDA mesh to legacy CPU TriangleMesh."""
    if isinstance(mesh_t, o3d.cuda.pybind.t.geometry.TriangleMesh) \
       or isinstance(mesh_t, o3d.t.geometry.TriangleMesh):
        return mesh_t.to_legacy()        # Open3D ≥0.15
    return mesh_t                        # already legacy


@torch.inference_mode()
def matched_object_ids(model, qg: SceneGraph, sg: SceneGraph):
    """
    Returns list of object IDs matched by either BigGNN or the pure-overlap fallback.
    """
    if model is None:
        print("No GNN model; using pure-overlap matcher")
        _, sub3d = get_matching_subgraph(qg, sg)
        return [] if sub3d is None else list(sub3d.nodes)
    device = next(model.parameters()).device
    def prep(g):
        n,e,f = g.to_pyg()
        return (
          torch.tensor(np.array(n), dtype=torch.float32, device=device),
          torch.tensor(np.array(e[0:2]),dtype=torch.int64,   device=device),
          torch.tensor(np.array(f),     dtype=torch.float32, device=device),
        )
    x_n,x_e,x_f = prep(qg)
    p_n,p_e,p_f = prep(sg)
    _ = model(x_n,p_n,x_e,p_e,x_f,p_f)    #TODO I think GNN output should not be ignored
    _, sub3d = get_matching_subgraph(qg, sg)
    # sub3d, _ = get_matching_subgraph(qg, sg) #TODO Which output to choose ? how to decide ?
    return [] if sub3d is None else list(sub3d.nodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan-id", required=True) # default="095821f7-e2c2-2de1-9568-b9ce59920e29"
    p.add_argument("--root",    required=True,
                   help="parent of 3RScan/<scan_id>/")
    p.add_argument("--graphs",  required=True,
                   help="folder with your .pt scene-graph files")
    p.add_argument("--ckpt",    default=None,
                   help="BigGNN checkpoint (.pt); omit for pure-overlap")
    p.add_argument("--N",       type=int, default=1)
    p.add_argument("--heads",   type=int, default=2)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    scan_dir  = Path(args.root) / args.scan_id
    graph_dir = Path(args.graphs)

    # load 3D scene-graph
    scenes = torch.load(graph_dir / "3dssg_graphs_processed_edgelists_relationembed.pt",
                        map_location=args.device)
    if args.scan_id not in scenes:
        raise ValueError(f"{args.scan_id} not found in 3dssg graphs")
    sg = SceneGraph(args.scan_id,
                    graph_type="3dssg",
                    graph=scenes[args.scan_id],
                    max_dist=1.0,
                    embedding_type="word2vec",
                    use_attributes=True)

    # load captions
    caps_raw = torch.load(graph_dir / "scanscribe_text_graphs_from_image_desc_node_edge_features.pt",
                          map_location=args.device)
    captions = {
      k: SceneGraph(args.scan_id,
                    graph_type="scanscribe",
                    graph=g,
                    embedding_type="word2vec",
                    use_attributes=True)
      for k,g in caps_raw.items() if k.startswith(args.scan_id)
    }
    if not captions:
        raise RuntimeError(f"No captions for {args.scan_id}")

    # optional BigGNN
    model = None
    if args.ckpt:
        if BigGNN is None:
            raise ImportError("BigGNN not available")
        ckpt = torch.load(args.ckpt, map_location=args.device)
        model = BigGNN(args.N, args.heads).to(args.device)
        model.load_state_dict(ckpt)
        model.eval()

    # load mesh + obj2faces
    mesh, obj2faces = load_scene(scan_dir)

    for cap_key, qg in captions.items():
        matched = matched_object_ids(model, qg, sg)
        # matched.append(12) #TODO for testing
        print(f"\n─────── {cap_key} ───────")
        print("Matched object IDs:", matched)
        print("SPACE/ENTER → next    |    q/Esc → quit")

        # fresh CPU copy via constructor
        cpu_mesh = to_legacy_cpu(mesh)          # guarantees legacy-CPU
        vis_mesh = colour_objects(cpu_mesh, obj2faces, matched)

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=cap_key, width=1280, height=720)
        vis.add_geometry(vis_mesh)
        vis.register_key_callback(32,  lambda v: v.close())  # SPACE
        vis.register_key_callback(257, lambda v: v.close())  # ENTER
        vis.register_key_callback(81,  lambda v: sys.exit(0))  # q
        vis.register_key_callback(113, lambda v: sys.exit(0))  # Q
        vis.register_key_callback(256, lambda v: sys.exit(0))  # ESC
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()



