# # ──────────────────────────────────  visualization.py  ──────────────────────────────────
# visualization.py
"""
For each human‐text caption, find its best‐matched scene, worst‐matched scene,
and its own (ground‐truth) scene, then show all three side‐by‐side with matched
objects highlighted.
Controls:
    SPACE / ENTER → next caption
    q / ESC       → quit
"""

import argparse, json, sys, random
from pathlib import Path

sys.path.append('../data_processing') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
sys.path.append('../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc/')

import numpy as np
import torch
import open3d as o3d
import torch.nn.functional as F

from scene_graph import SceneGraph
from data_distribution_analysis.helper import get_matching_subgraph
from model_graph2graph import BigGNN

def load_scene(scan_dir: Path):
    """Reads labels.instances.annotated.v2.ply and returns (legacy‐CPU mesh, obj2faces)."""
    ply = scan_dir / "labels.instances.annotated.v2.ply"
    if not ply.exists():
        raise FileNotFoundError(f"Missing mesh file: {ply}")
    mesh = o3d.io.read_triangle_mesh(str(ply))
    mesh.compute_vertex_normals()
    try:
        mesh = mesh.cpu()
    except AttributeError:
        pass  # already CPU

    # pack per-vertex RGB -> 24-bit ints
    vc   = np.asarray(mesh.vertex_colors)
    vc8  = (vc * 255 + 0.5).astype(np.uint32)
    vhex = (vc8[:,0] << 16) | (vc8[:,1] << 8) | vc8[:,2]

    # load objects.json (one level up)
    objj = scan_dir.parent / "objects.json"
    scans = {s["scan"]: s for s in json.load(open(objj))["scans"]}
    meta  = scans.get(scan_dir.name)
    if meta is None:
        raise KeyError(f"{scan_dir.name} not in {objj}")
    color2oid = {int(o["ply_color"].lstrip("#"),16): int(o["id"])
                 for o in meta["objects"]}

    # vertex -> object ID
    v_oid = np.array([ color2oid.get(int(h),0) for h in vhex ], dtype=np.int64)

    # facewise majority vote
    tris  = np.asarray(mesh.triangles)
    f_oid = np.array([ np.bincount(v_oid[t]).argmax() for t in tris ],
                     dtype=np.int64)

    # invert -> obj2faces
    obj2faces = {}
    for fid, oid in enumerate(f_oid):
        if oid == 0: continue
        obj2faces.setdefault(int(oid), []).append(fid)
    obj2faces = {k: np.array(v, dtype=np.int64) for k,v in obj2faces.items()}
    if not obj2faces:
        raise RuntimeError(f"No labeled faces in {scan_dir.name}")

    return mesh, obj2faces

def colour_objects(mesh, obj2faces, matched, base=(0.6,0.6,0.6)):
    """Paint every vertex grey, then each `matched` object bright random color."""
    n_v   = len(mesh.vertices)
    cols  = np.tile(base, (n_v,1))
    rng   = np.random.default_rng(42)
    tris  = np.asarray(mesh.triangles)

    for oid in matched:
        tri_ids = obj2faces.get(oid, [])
        c       = rng.random(3)
        for tid in tri_ids:
            for vid in tris[tid]:
                cols[int(vid)] = c

    mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
    return mesh

@torch.inference_mode()
def compute_match_score(model, qg: SceneGraph, sg: SceneGraph, device: str):
    """
    Runs BigGNN on the *matched* subgraphs of (qg, sg).  
    Falls back to full‐graph DBSCAN overlap if subgraphs are degenerate.
    Returns the average of matching probability m_p and normalized cosine‐sim.
    """
    # 1) Extract the two subgraphs via DBSCAN overlap
    q_sub, s_sub = get_matching_subgraph(qg, sg)

    # 2) If either subgraph is too small (or None), fall back to full graphs
    def is_degenerate(g: SceneGraph):
        return (g is None or
                len(g.nodes) <= 1 or
                (hasattr(g, 'edge_idx') and len(g.edge_idx[0]) < 1))
    if is_degenerate(q_sub) or is_degenerate(s_sub):
        print(f"DEGENERATE GRAPH: falling back to full graph match")
        q_sub, s_sub = qg, sg

    # 3) Prepare PyG inputs
    def prep(g: SceneGraph):
        n, e, f = g.to_pyg()
        return (
            torch.tensor(np.array(n), dtype=torch.float32, device=device),
            torch.tensor(np.array(e[0:2]), dtype=torch.int64,   device=device),
            torch.tensor(np.array(f),      dtype=torch.float32, device=device),
        )

    x_n, x_e, x_f = prep(q_sub)
    p_n, p_e, p_f = prep(s_sub)

    # 4) Forward pass through the model
    x_p, p_p, m_p = model(x_n, p_n, x_e, p_e, x_f, p_f)   # x_p, p_p are embeddings

    # 5) Compute cosine‐similarity and normalize to [0,1]
    cos_sim = F.cosine_similarity(x_p, p_p, dim=0).item()  # in [-1,1]
    cos_sim = (cos_sim + 1.0) / 2.0                        # now in [0,1]

    match_prob = m_p.item()                                # in [0,1]

    # 6) Return their average (or any weighting you prefer)
    return 0.5 * match_prob + 0.5 * cos_sim


def visualize_match(scan_root: Path,
                    qg: SceneGraph,
                    sg: SceneGraph):
    """
    Visualize a single query‐graph qg vs one scene‐graph sg:
      - loads the mesh for sg.scene_id
      - finds the matched object IDs via DBSCAN overlap
      - colours them and pops up the window
    Controls:
      SPACE/ENTER → close window
      q / ESC     → quit program
    """
    # 1) find matched object‐IDs via the pure‐overlap subgraph
    _, sub3d = get_matching_subgraph(qg, sg)
    matched = list(sub3d.nodes) if sub3d else []

    # 2) load and colour the mesh
    mesh, obj2faces = load_scene(scan_root / sg.scene_id)
    vis_mesh = colour_objects(mesh, obj2faces, matched)

    # 3) print info
    print(f"\n>>> Query scene {qg.scene_id} vs scene {sg.scene_id}")
    print("Matched object IDs:", matched)
    print(" SPACE/ENTER → close window;  q/Esc → quit")

    # 4) show it
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"{qg.scene_id}→{sg.scene_id}", width=1024, height=768)
    vis.add_geometry(vis_mesh)
    vis.register_key_callback(32,  lambda v: v.close())   # SPACE
    vis.register_key_callback(257, lambda v: v.close())   # ENTER
    vis.register_key_callback(81,  lambda v: sys.exit(0)) # q
    vis.register_key_callback(256, lambda v: sys.exit(0)) # ESC
    vis.run()
    vis.destroy_window()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",   required=True,
                   help="parent folder of 3RScan/<scan_id>/")
    p.add_argument("--graphs", required=True,
                   help="folder with your processed .pt graphs")
    p.add_argument("--ckpt",   default=None,
                   help="BigGNN checkpoint (.pt); omit for DBSCAN only")
    p.add_argument("--N",      type=int, default=1)
    p.add_argument("--heads",  type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # 1) load scene-graphs
    raw3d = torch.load(Path(args.graphs)/"3dssg"/"3dssg_graphs_processed_edgelists_relationembed.pt",
                       map_location="cpu")
    # raw3d = torch.load(Path(args.graphs)/"3dssg"/"3dssg_graphs_processed_edgelists_relationembed.pt",
    #                    map_location="cpu")
    database_3dssg = {
      sid: SceneGraph(sid, graph_type="3dssg", graph=g,
                      max_dist=1.0, embedding_type="word2vec",
                      use_attributes=True)
      for sid,g in raw3d.items()
    }

    # 2) load text-graphs
    rawtxt = torch.load(Path(args.graphs)/"scanscribe"/"scanscribe_text_graphs_from_image_desc_node_edge_features.pt", map_location="cpu")
    # rawtxt = torch.load(Path(args.graphs)/"human"/"human_graphs_processed.pt", map_location="cpu")
    dataset = [
      SceneGraph(k.split("_")[0], txt_id=None,
                 graph=g, graph_type="human",
                 embedding_type="word2vec", use_attributes=True)
      for k,g in rawtxt.items()
    ]

    # 3) optional GNN
    model = None
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=args.device)
        model = BigGNN(args.N, args.heads).to(args.device)
        model.load_state_dict(ckpt)
        model.eval()

    # 4) for each caption → find best, worst, and gt scenes, then show them in order
    for qg in dataset:
        # compute scores over all scenes
        scores = {
          sid: compute_match_score(model, qg, sg, args.device)
          for sid, sg in database_3dssg.items()
        }
        best_sid  = max(scores,  key=scores.get)
        worst_sid = min(scores,  key=scores.get)
        gt_sid    = qg.scene_id

        print("\n" + "="*60)
        print(f"Query caption for scene: {gt_sid}")
        print(f" → Best match:  {best_sid}  (score={scores[best_sid]:.4f})")
        print(f" → Worst match: {worst_sid} (score={scores[worst_sid]:.4f})")
        print(f" → GroundTruth: {gt_sid}  (score={scores[gt_sid]:.4f})")

        # 5) visualize each separately
        visualize_match(Path(args.root), qg, database_3dssg[best_sid])
        visualize_match(Path(args.root), qg, database_3dssg[worst_sid])
        visualize_match(Path(args.root), qg, database_3dssg[gt_sid])


if __name__ == "__main__":
    main()

# # ────────────────────────────────────────────────────────────────────────────────