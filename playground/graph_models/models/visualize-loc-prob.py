#!/usr/bin/env python3
"""
visualize-loc-prob.py  ·  dense-grid version  ·  June 2025
-----------------------------------------------------------

For every ScanScribe caption graph:

1.  Load its ground-truth 3D-SSG scene-graph + mesh.
2.  Compute cosine similarity between all nodes → keep Top-K object matches.
3.  Generate a dense XY grid (spacing = --grid_step) at eye-height.
4.  For each grid cell (≙ candidate camera) cast one ray to every matched
    object centroid; count how many objects are the *first* hit.
5.  Convert counts → posterior probabilities.
6.  Visualise
      • probability heat-map (optional, 2-D)  
      • full 3-D mesh with matched objects bright and camera spheres whose
        colour encodes probability (optional, Open3D).

Author: VLN

python visualize-loc-prob.py     --root  /home/klrshak/work/VisionLang/3RScan/data/3RScan     --graphs /home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data --top_k 5 --show_heatmap --show_3d
"""

from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c

# --------------------------------------------------------------------------- #
#  Repo imports (same trick as visualization_graph-object.py)                 #
# --------------------------------------------------------------------------- #

sys.path.append('../data_processing') # sys.path.append('/home/julia/Documents/h_coarse_loc/playground/graph_models/data_processing')
sys.path.append('../../../') # sys.path.append('/home/julia/Documents/h_coarse_loc/')

from scene_graph import SceneGraph                    # noqa: E402
from data_distribution_analysis.helper import get_matching_subgraph  # noqa: E402

# ════════════════════════════════════════════════════════════════════════════
#  Utility: load mesh + obj↔face maps (verbatim copy, trimmed)               ═
# ════════════════════════════════════════════════════════════════════════════
def load_scene(scan_dir: Path):
    """Return (legacy mesh, faces→object-id array, obj→faces dict)."""
    ply = scan_dir / "labels.instances.annotated.v2.ply"
    if not ply.exists():
        raise FileNotFoundError(ply)
    mesh = o3d.io.read_triangle_mesh(str(ply))
    mesh.compute_vertex_normals()

    vc   = (np.asarray(mesh.vertex_colors) * 255 + 0.5).astype(np.uint32)
    vhex = (vc[:, 0] << 16) | (vc[:, 1] << 8) | vc[:, 2]

    meta = {
        s["scan"]: s
        for s in json.load(open(scan_dir.parent / "objects.json"))["scans"]
    }[scan_dir.name]
    color2oid = {int(o["ply_color"].lstrip("#"), 16): int(o["id"])
                 for o in meta["objects"]}

    v_oid = np.array([color2oid.get(int(h), 0) for h in vhex], dtype=np.int32)
    tris  = np.asarray(mesh.triangles, dtype=np.int32)
    tri2obj = np.array([np.bincount(v_oid[t]).argmax() for t in tris],
                       dtype=np.int32)

    obj2faces = {}
    for fid, oid in enumerate(tri2obj):
        if oid != 0:
            obj2faces.setdefault(int(oid), []).append(fid)
    obj2faces = {k: np.asarray(v, dtype=np.int32) for k, v in obj2faces.items()}
    return mesh, tri2obj, obj2faces


# ════════════════════════════════════════════════════════════════════════════
#  Cosine-similarity Top-K matcher                                            ═
# ════════════════════════════════════════════════════════════════════════════
def topk_matched_objects(qg: SceneGraph, sg: SceneGraph, k: int = 5):
    qf, _, _ = qg.to_pyg()
    sf, _, _ = sg.to_pyg()
    qf = F.normalize(torch.tensor(np.asarray(qf), dtype=torch.float32), dim=1)
    sf = F.normalize(torch.tensor(np.asarray(sf), dtype=torch.float32), dim=1)

    sim = qf @ sf.T                                   # (|Q|, |S|)
    topv, topi = torch.topk(sim.flatten(), min(k, sim.numel()))
    sids  = list(sg.nodes)
    S     = sf.size(0)
    picks = []
    for idx in topi.tolist():
        sid = sids[idx % S]
        if sid not in picks:
            picks.append(sid)
        if len(picks) == k:
            break
    return picks


# ════════════════════════════════════════════════════════════════════════════
#  Camera grid sampler                                                        ═
# ════════════════════════════════════════════════════════════════════════════
def sample_grid(verts: np.ndarray, step: float, z_eye: float = 1.6):
    """
    Return (N,3) xyz grid points covering the mesh’s XY AABB, spacing = step.
    """
    xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]
    gx = np.arange(xs.min(), xs.max() + 1e-4, step)
    gy = np.arange(ys.min(), ys.max() + 1e-4, step)
    xv, yv = np.meshgrid(gx, gy, indexing="xy")
    n = xv.size
    cams = np.stack([xv.ravel(), yv.ravel(), np.full(n, zs.min() + z_eye)],
                    axis=1)
    return cams


# ════════════════════════════════════════════════════════════════════════════
#  Visibility test (single ray)                                               ═
# ════════════════════════════════════════════════════════════════════════════
def first_hit_is_object(cam: np.ndarray, centre: np.ndarray, target_oid: int,
                        rc: o3d.t.geometry.RaycastingScene,
                        tri2obj: np.ndarray) -> bool:
    d = centre - cam
    l = np.linalg.norm(d)
    if l < 1e-6:
        return False
    ray = np.concatenate([cam, d / l])[None, :]
    ans = rc.cast_rays(o3c.Tensor(ray, dtype=o3c.Dtype.Float32))
    tri = int(ans["primitive_ids"].cpu().numpy()[0])
    if tri < 0 or tri >= len(tri2obj):
        return False
    return int(tri2obj[tri]) == int(target_oid)


# ════════════════════════════════════════════════════════════════════════════
#  Colour helpers for visualising                                             ═
# ════════════════════════════════════════════════════════════════════════════
def colour_objects(mesh: o3d.geometry.TriangleMesh,
                   obj2faces: dict[int, np.ndarray],
                   focus: list[int]):
    """Grey base mesh; bright random colour for every object in `focus`."""
    rng = np.random.default_rng(42)
    grey = np.full((len(mesh.vertices), 3), 0.55)
    tris = np.asarray(mesh.triangles)
    for oid in focus:
        for fid in obj2faces.get(oid, []):
            for vid in tris[fid]:
                grey[int(vid)] = rng.random(3)
    mesh.vertex_colors = o3d.utility.Vector3dVector(grey)
    return mesh


def colormap(vals: np.ndarray):
    """Map values in [0,1] → RGB using matplotlib’s 'hot'."""
    cmap = plt.get_cmap("hot")
    return cmap(vals)[:, :3]


# ════════════════════════════════════════════════════════════════════════════
#  Main programme                                                             ═
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Compute and visualise localisation probability surface "
                    "for ScanScribe captions.")
    parser.add_argument("--root", required=True,
                        help="Parent folder of 3RScan/<scan_id>/")
    parser.add_argument("--graphs", required=True,
                        help="processed_data/{3dssg,scanscribe}/")
    parser.add_argument("--top_k", type=int, default=25,
                        help="How many object matches to keep per caption")
    parser.add_argument("--grid_step", type=float, default=0.25,
                        help="XY grid spacing in metres")
    parser.add_argument("--query_limit", type=int,
                        help="Process only the first N captions (debug)")
    parser.add_argument("--show_heatmap", action="store_true",
                        help="Show 2-D Matplotlib heat-map")
    parser.add_argument("--show_3d", action="store_true",
                        help="Open Open3D viewer with mesh + probability spheres")
    args = parser.parse_args()

    # ----- quick summary of chosen arguments
    print("\nConfiguration -------------------------------------------")
    for k, v in vars(args).items():
        print(f"  {k:<12}: {v}")
    print("---------------------------------------------------------\n")

    # ----- load scene graphs ----------------------------------------------
    g3d = torch.load(Path(args.graphs) / "3dssg" /
                     "3dssg_graphs_processed_edgelists_relationembed.pt",
                     map_location="cpu")
    scenes = {sid: SceneGraph(sid,
                              graph_type="3dssg",
                              graph=g,
                              max_dist=1.0,        # ← FIX
                              embedding_type="word2vec",
                              use_attributes=True)
              for sid, g in g3d.items()}

    gtxt = torch.load(Path(args.graphs) / "scanscribe" /
                      "scanscribe_text_graphs_from_image_desc_node_edge_features.pt",
                      map_location="cpu")
    queries = [SceneGraph(k.split("_")[0],
                          txt_id=None,
                          graph=g,
                          graph_type="scanscribe",
                          embedding_type="word2vec",
                          use_attributes=True)
               for k, g in gtxt.items()]
    if args.query_limit:
        queries = queries[: args.query_limit]

    # ----- iterate over captions ------------------------------------------
    for qi, qg in enumerate(queries, 1):
        sid = qg.scene_id
        sg  = scenes[sid]

        obj_ids = topk_matched_objects(qg, sg, k=args.top_k)
        if not obj_ids:
            print(f"[{qi}] {sid} : no cosine matches — skipped")
            continue

        # mesh + ray-caster
        mesh, tri2obj, obj2faces = load_scene(Path(args.root) / sid)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        verts = np.asarray(mesh.vertices)
        cams  = sample_grid(verts, step=args.grid_step)
        print(f"[{qi}] {sid}: grid {len(cams):,} pts  |  {len(obj_ids)} objs")

        # object centroids
        tris = np.asarray(mesh.triangles)
        centroids = {}
        for oid in obj_ids:
            faces = obj2faces.get(oid)                # <- changed
            if faces is not None and len(faces):      # <- changed
                centroids[oid] = verts[np.unique(tris[faces].ravel())].mean(0)


        # visibility
        scores = np.zeros(len(cams), dtype=np.int32)
        for idx, cam in enumerate(cams):
            for oid, cen in centroids.items():
                if first_hit_is_object(cam, cen, oid, rc, tri2obj):
                    scores[idx] += 1
        if scores.sum() == 0:
            print("    none of the matched objects visible — skipped\n")
            continue
        probs = scores / scores.sum()

        # --- 2-D heat-map
        if args.show_heatmap:
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(cams[:, 0], cams[:, 1], c=probs,
                             cmap="hot", s=12)
            plt.colorbar(sc, label="probability")
            plt.title(f"{sid}  –  grid {args.grid_step} m")
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

        # --- 3-D viewer
        if args.show_3d:
            vis_mesh = colour_objects(mesh, obj2faces, obj_ids)
            spheres  = []
            for p, col in zip(cams, colormap(probs)):
                s = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                s.translate(p);  s.paint_uniform_color(col);  spheres.append(s)

            vis = o3d.visualization.Visualizer()
            vis.create_window(width=1280, height=800,
                              window_name=f"{sid} – localisation prob.")
            vis.add_geometry(vis_mesh)
            for s in spheres:
                vis.add_geometry(s)
            vis.get_render_option().point_size = 3
            vis.run();  vis.destroy_window()


if __name__ == "__main__":
    main()