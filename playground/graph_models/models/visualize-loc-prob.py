"""
visualize-loc-prob.py
=====================

For *each* ScanScribe caption graph:

1. Load its ground-truth 3D-SSG scene-graph and mesh.
2. Compute cosine similarity between every text node and every scene node;
   pick the **top-K** scene objects as putative matches (default K=5).
3. Build a 36-particle ring (uniform 10° steps, omnidirectional camera) around
   the scene at pedestrian eye-height.
4. For every particle, cast a single ray towards the centroid of each matched
   object using Open3D's `RaycastingScene`.  An object counts as *visible* if
   it is the **first** triangle hit by that ray.
5. A particle’s **score** is the number of visible matched objects ∈ {0…K}.
   Normalise the scores → posterior probability `p(x | caption)`.
6. Optionally visualise the probability field with Matplotlib.

This is a one-shot localisation likelihood, **not** a full particle-filter
time series (see earlier discussion).

Author: VLN · June 2025
"""

from __future__ import annotations
import argparse, sys, json, math, itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import open3d.core as o3c               # for tgeometry ray-casting
import matplotlib.pyplot as plt         # optional heat-map

sys.path.append('../data_processing')
sys.path.append('../../../')

from scene_graph import SceneGraph
from model_graph2graph import BigGNN    # only needed for SceneGraph.to_pyg()
from data_distribution_analysis.helper import get_matching_subgraph
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Helpers                                                                     │
# ──────────────────────────────────────────────────────────────────────────────
def load_scene(scan_dir: Path):
    """
    **Copied & lightly modified** from visualization_graph-object.py.
    Returns:
        mesh_legacy      – o3d.geometry.TriangleMesh
        tris_np          – (F,3) numpy ints
        tri2obj          – np.int32[F] mapping triangle-index → object-ID
        obj2faces        – dict[objID] → np.ndarray[face_ids]
    """
    ply = scan_dir / "labels.instances.annotated.v2.ply"
    if not ply.exists():
        raise FileNotFoundError(ply)
    mesh = o3d.io.read_triangle_mesh(str(ply))
    mesh.compute_vertex_normals()

    vc8 = (np.asarray(mesh.vertex_colors) * 255 + 0.5).astype(np.uint32)
    vhex = (vc8[:, 0] << 16) | (vc8[:, 1] << 8) | vc8[:, 2]

    meta = {
        s["scan"]: s
        for s in json.load(open(scan_dir.parent / "objects.json"))["scans"]
    }[scan_dir.name]
    color2oid = {int(o["ply_color"].lstrip("#"), 16): int(o["id"])
                 for o in meta["objects"]}

    v_oid = np.array([color2oid.get(int(h), 0) for h in vhex], dtype=np.int32)
    tris  = np.asarray(mesh.triangles, dtype=np.int32)
    tri2obj = np.array(
        [np.bincount(v_oid[t]).argmax() for t in tris], dtype=np.int32
    )

    obj2faces = {}
    for fid, oid in enumerate(tri2obj):
        if oid == 0:
            continue
        obj2faces.setdefault(int(oid), []).append(fid)
    obj2faces = {k: np.asarray(v, dtype=np.int32) for k, v in obj2faces.items()}

    return mesh, tris, tri2obj, obj2faces


def topk_matched_objects(qg: SceneGraph,
                         sg: SceneGraph,
                         k: int = 5):
    """
    Returns up to *k* scene object-IDs whose node embeddings have the highest
    cosine similarity to *any* text-node embedding.
    """
    q_feat, _, _ = qg.to_pyg()
    s_feat, _, _ = sg.to_pyg()
    q_feat = torch.tensor(np.asarray(q_feat), dtype=torch.float32)
    s_feat = torch.tensor(np.asarray(s_feat), dtype=torch.float32)
    q_feat = F.normalize(q_feat, dim=1)
    s_feat = F.normalize(s_feat, dim=1)

    sim = torch.matmul(q_feat, s_feat.T)          # (|Q|, |S|)
    flat = torch.flatten(sim)
    vals, idx = torch.topk(flat, min(k, flat.numel()))
    s_ids = list(sg.nodes)                        # order matches to_pyg()
    s_size = s_feat.size(0)

    picked = []
    for f_idx in idx.tolist():
        s_idx = f_idx % s_size
        oid = s_ids[s_idx]
        if oid not in picked:
            picked.append(oid)
        if len(picked) == k:
            break
    return picked


def sample_camera_positions(verts_xyz: np.ndarray,
                            grid_step: float,
                            ring_n: int = 36,
                            ring_radius_factor: float = 1.05,
                            z_eye: float = 1.6) -> np.ndarray:
    """
    verts_xyz : (V,3) array of mesh vertices
    ...
    """
    xs, ys, zs = verts_xyz[:, 0], verts_xyz[:, 1], verts_xyz[:, 2]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    z_floor      = zs.min()

    if grid_step > 0:                                   # dense grid
        gx = np.arange(min_x, max_x + 1e-4, grid_step)
        gy = np.arange(min_y, max_y + 1e-4, grid_step)
        xv, yv = np.meshgrid(gx, gy, indexing='xy')
        n = xv.size
        cam = np.stack([xv.ravel(), yv.ravel(),
                        np.full(n, z_floor + z_eye)], axis=1)
        return cam

    # ring fallback
    centre = np.array([(min_x + max_x) * 0.5,
                       (min_y + max_y) * 0.5])
    radius = ring_radius_factor * np.linalg.norm(
                 verts_xyz[:, :2] - centre, axis=1).max()
    angles = np.linspace(0, 2 * math.pi, ring_n, endpoint=False)
    xr = centre[0] + radius * np.cos(angles)
    yr = centre[1] + radius * np.sin(angles)
    zr = np.full_like(xr, z_floor + z_eye)
    return np.stack([xr, yr, zr], axis=1)



def visible(camera_pos: np.ndarray,
            obj_centroid: np.ndarray,
            target_oid: int,
            rc_scene: o3d.t.geometry.RaycastingScene,
            tri2obj: np.ndarray) -> bool:
    """
    Cast one ray from camera → object centroid.
    Return True iff the first hit triangle belongs to `target_oid`.
    """
    d = obj_centroid - camera_pos
    norm = np.linalg.norm(d)
    if norm < 1e-6:
        return False
    d /= norm

    ray = o3c.Tensor(np.concatenate([camera_pos, d])[None, :],
                     dtype=o3c.Dtype.Float32)
    ans = rc_scene.cast_rays(ray)
    tri_raw = int(ans['primitive_ids'].cpu().numpy()[0])  # -1 or uint32_t max on miss
    if tri_raw < 0 or tri_raw >= len(tri2obj):
        return False                                      # miss
    return int(tri2obj[tri_raw]) == int(target_oid)


# ──────────────────────────────────────────────────────────────────────────────
# Main                                                                         │
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",   required=True,
                    help="parent folder of 3RScan/<scan_id>/")
    ap.add_argument("--graphs", required=True,
                    help="folder with processed_data/{3dssg,scanscribe}/")
    ap.add_argument("--top_k",  type=int, default=5,
                    help="how many object matches to keep per caption")
    ap.add_argument("--particles", type=int, default=36,
                    help="number of candidate positions on the ring")
    ap.add_argument("--query_limit", type=int, default=None,
                    help="only show the first N captions (debug)")
    ap.add_argument("--show_plot", action="store_true",
                    help="plot heat-map using matplotlib")
    ap.add_argument("--grid_step", type=float, default=0.2,
                help="Grid spacing in metres (0 → use ring method)")
    args = ap.parse_args()

    # 1. Load graphs (same helper paths as in the original visualiser)
    raw3d = torch.load(Path(args.graphs)/"3dssg"/
                       "3dssg_graphs_processed_edgelists_relationembed.pt",
                       map_location="cpu")
    database = {
        sid: SceneGraph(sid, graph_type="3dssg", graph=g,
                        max_dist=1.0, embedding_type="word2vec",
                        use_attributes=True)
        for sid, g in raw3d.items()
    }

    rawtxt = torch.load(Path(args.graphs)/"scanscribe"/
                        "scanscribe_text_graphs_from_image_desc_node_edge_features.pt",
                        map_location="cpu")
    queries = [
        SceneGraph(k.split("_")[0], txt_id=None,
                   graph=g, graph_type="scanscribe",
                   embedding_type="word2vec", use_attributes=True)
        for k, g in rawtxt.items()
    ]
    if args.query_limit:
        queries = queries[:args.query_limit]

    # 2. Process each caption
    for qi, qg in enumerate(queries, 1):
        scene_id = qg.scene_id
        sg = database[scene_id]

        # 2.1 object matches
        obj_ids = topk_matched_objects(qg, sg, k=args.top_k)
        if not obj_ids:
            print(f"[{qi}] {scene_id}: no matches – skipped")
            continue

        # 2.2 load mesh & build ray-caster
        scan_dir = Path(args.root) / scene_id
        mesh_legacy, tris_np, tri2obj_np, obj2faces = load_scene(scan_dir)
        tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
        rscene = o3d.t.geometry.RaycastingScene()
        rscene.add_triangles(tmesh)

        # 2.3 object centroids
        tris = tris_np
        verts = np.asarray(mesh_legacy.vertices)
        centroids = {}
        for oid in obj_ids:
            faces = obj2faces.get(oid)
            if faces is None or len(faces) == 0:
                continue
            v_id = tris[faces].reshape(-1)
            centroids[oid] = verts[np.unique(v_id)].mean(0)

        # 2.4 sample particles
        verts_xy = verts[:, :2]
        cam_pos = sample_camera_positions(
            verts,
            grid_step=args.grid_step,
            ring_n=args.particles)

        print(f"    sampled {len(cam_pos):,} candidate viewpoints "
                f"({'grid' if args.grid_step > 0 else 'ring'})")

        # 2.5 visibility test
        scores = np.zeros(len(cam_pos), dtype=np.int32)
        for pi, pos in enumerate(cam_pos):
            for oid, c in centroids.items():
                if visible(pos, c, oid, rscene, tri2obj_np):
                    scores[pi] += 1



        if scores.sum() == 0:
            print(f"[{qi}] {scene_id}: none of the {args.top_k} objects visible "
                  "from ring – skipped")
            continue

        probs = scores / scores.sum()

        # 2.6 report / visualise
        print(f"\n[{qi}/{len(queries)}] Scene {scene_id}")
        for i, p in enumerate(probs):
            # print(f"  particle {i:02d}: score={scores[i]}  prob={p:.3f}")

        if args.show_plot:
            plt.figure(figsize=(6, 6))
            plt.title(f"p(x | caption) – {scene_id}")
            sc = plt.scatter(cam_pos[:, 0], cam_pos[:, 1],
                             c=probs, cmap='hot', s=80)
            plt.colorbar(sc, label='probability')
            plt.axis('equal')
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()



"""
:Visualize-localization-probability:

Algorithm:
1. We obtain the scene graph and the text graph.
2. Calculate cosine similarity between all nodes in the scene and text graphs.
3. Threshold the similarities to obtain top 5 potential matches.
4. Load the 3D meshes of the scene.
5. For Loop through the top 5 matches:
    5.1 For each match, find the object instances from text-graph in the 3D mesh.
    5.2 intantiate 36 possible particles/positions uniformly (360 FoV camera) in the 3D mesh scene.
    5.3 At each possible postion, check which particles can see, each object instance (using ray-casting)
    5.4 Give each possible particle a score based on how many objects it can see.
    5.5 Create a probability heatmatrix of the particles based on the scores.

    
Take Reference from visualization_graph-object.py to know how to find match scene and text graph and also find common objects between them.
"""

