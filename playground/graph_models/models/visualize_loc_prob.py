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



Adds a second 2-D plot: a quiver of *weighted arrows* whose:
  • direction = average of casted ray directions (unit vectors) that fit in a
    camera FOV window (H×V) which *maximizes* the number of visible objects.
  • size     = that maximal count (scaled).
                

Author: Shak

python visualize-loc-prob.py     --root  /home/klrshak/work/VisionLang/3RScan/data/3RScan     --graphs /home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data --top_k 5 --show_heatmap --show_3d

New CLI flags:
  --show_arrows         Show weighted-arrow (quiver) plot
  --h_fov_deg 100       Horizontal FOV (degrees)
  --v_fov_deg 60        Vertical FOV (degrees)
  --arrow_stride 2      Plot every Nth grid camera (reduce clutter)
  --arrow_len 0.0       Max arrow length in metres (0 → 0.9*grid_step)

The rest of the pipeline (heatmap, Open3D) is unchanged.
"""

from __future__ import annotations
import argparse, hashlib, json, math, sys
from pathlib import Path
import re

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open3d as o3d
import open3d.core as o3c

# --------------------------------------------------------------------------- #
#  Repo imports                                                               #
# --------------------------------------------------------------------------- #

sys.path.append('../data_processing')
sys.path.append('../../../')

from scene_graph import SceneGraph                    # noqa: E402
from graph_loader_utils import get_word2vec          # noqa: E402
from data_distribution_analysis.helper import get_matching_subgraph  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════
#  Utility: load mesh + obj↔face maps                                         ═
# ════════════════════════════════════════════════════════════════════════════
def load_scene(scan_dir: Path, dataset: str = "3rscan"):
    """Return (legacy mesh, faces→object-id array, obj→faces dict).

    Backward-compatible wrapper used by evaluation scripts. Defaults to the
    historic 3RScan behavior when dataset is not provided.
    """
    from localization_dataset_io import load_scene_geometry

    return load_scene_geometry(scan_dir, dataset=dataset)


# ════════════════════════════════════════════════════════════════════════════
#  Cosine-similarity Top-K matcher                                            ═
# ════════════════════════════════════════════════════════════════════════════
def topk_matched_objects(qg: SceneGraph,
                         sg: SceneGraph,
                         k: int = 5,
                         return_scores: bool = False,
                         use_subgraph: bool = False,
                         score_threshold: float = -1.0,
                         dynamic_k: bool = False,
                         exact_label_score: float = -1e9,
                         homogenize_label_embeddings: bool = False,
                         ensure_query_coverage: bool = False):
    query_obj_count = len(qg.nodes)
    if dynamic_k:
        k = query_obj_count
    k = max(1, int(k))

    if use_subgraph:
        try:
            q_sub, s_sub = get_matching_subgraph(qg, sg)
        except Exception:  # noqa: BLE001
            q_sub, s_sub = None, None
        if q_sub is None or len(q_sub.nodes) < 3:
            q_sub = qg
        if s_sub is None or len(s_sub.nodes) < 3:
            s_sub = sg
        qg = q_sub
        sg = s_sub

    qids = list(qg.nodes)
    sids = list(sg.nodes)
    if not qids or not sids:
        if return_scores:
            return [], []
        return []

    if homogenize_label_embeddings:
        qmat = np.asarray([label_embedding_for_matching(qg.nodes[qid].label)
                           for qid in qids], dtype=np.float32)
        smat = np.asarray([label_embedding_for_matching(sg.nodes[sid].label)
                           for sid in sids], dtype=np.float32)
    else:
        qmat = np.asarray([qg.nodes[qid].features for qid in qids], dtype=np.float32)
        smat = np.asarray([sg.nodes[sid].features for sid in sids], dtype=np.float32)

    qf = F.normalize(torch.tensor(qmat, dtype=torch.float32), dim=1)
    sf = F.normalize(torch.tensor(smat, dtype=torch.float32), dim=1)
    if qf.numel() == 0 or sf.numel() == 0:
        if return_scores:
            return [], []
        return []

    sim = qf @ sf.T                                   # (|Q|, |S|)
    qlabels = [qg.nodes[qid].label for qid in qids]
    slabels = [sg.nodes[sid].label for sid in sids]
    sim = apply_exact_label_score(sim, qlabels, slabels, exact_label_score)
    picks = []
    scores = []
    picked_set = set()
    S = sf.size(0)

    if ensure_query_coverage:
        # Pass 1: each query node contributes its best available match.
        for qi in range(sim.size(0)):
            row_order = torch.argsort(sim[qi], descending=True)
            for si in row_order.tolist():
                val = float(sim[qi, si])
                if val < float(score_threshold):
                    break
                sid = sids[si]
                if sid in picked_set:
                    continue
                picks.append(sid)
                scores.append(val)
                picked_set.add(sid)
                break
            if len(picks) == k:
                break

    if len(picks) < k:
        # Pass 2: fill remaining slots from global top scores.
        flat = sim.flatten()
        if flat.numel() > 0:
            order = torch.argsort(flat, descending=True)
            for idx in order.tolist():
                val = float(flat[idx])
                if val < float(score_threshold):
                    break
                sid = sids[idx % S]
                if sid in picked_set:
                    continue
                picks.append(sid)
                scores.append(val)
                picked_set.add(sid)
                if len(picks) == k:
                    break
    if return_scores:
        return picks, scores
    return picks


def canonical_label(label: str) -> str:
    """Light canonicalization for label identity matching."""
    text = str(label).strip().lower()
    text = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_MATCH_W2V_CACHE: dict[str, np.ndarray] = {}
_MATCH_EMBED_CACHE: dict[str, np.ndarray] = {}
_MATCH_EMBED_DIM = 300
_MATCH_W2V_BLEND = 0.8


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if not np.isfinite(nrm) or nrm < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / nrm).astype(np.float32)


def _stable_hash32(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _hashed_label_embedding(key: str, dim: int = _MATCH_EMBED_DIM) -> np.ndarray:
    """Deterministic lexical embedding used when word2vec is weak/OOV."""
    out = np.zeros(dim, dtype=np.float32)
    if not key:
        return out
    wrapped = f"^{key}$"
    for n in (3, 4, 5):
        if len(wrapped) < n:
            continue
        for i in range(len(wrapped) - n + 1):
            gram = wrapped[i:i + n]
            h = _stable_hash32(gram)
            out[h % dim] += 1.0 if ((h >> 1) & 1) == 0 else -1.0
    if not np.any(out):
        out[_stable_hash32(wrapped) % dim] = 1.0
    return _l2_normalize(out)


def label_embedding_for_matching(label: str) -> np.ndarray:
    """Shared label encoding for query/scene matching with robust OOV fallback."""
    key = canonical_label(label)
    cached = _MATCH_EMBED_CACHE.get(key)
    if cached is not None:
        return cached

    lexical = _hashed_label_embedding(key, dim=_MATCH_EMBED_DIM)
    w2v = get_word2vec(key, _MATCH_W2V_CACHE)
    vec = w2v[0] if isinstance(w2v, tuple) else w2v
    w2v_arr = np.asarray(vec, dtype=np.float32).reshape(-1)

    if w2v_arr.size == 0:
        w2v_arr = np.zeros(_MATCH_EMBED_DIM, dtype=np.float32)
    elif w2v_arr.size > _MATCH_EMBED_DIM:
        w2v_arr = w2v_arr[:_MATCH_EMBED_DIM]
    elif w2v_arr.size < _MATCH_EMBED_DIM:
        w2v_arr = np.pad(w2v_arr, (0, _MATCH_EMBED_DIM - w2v_arr.size))

    w2v_unit = _l2_normalize(w2v_arr)
    if not np.any(w2v_unit):
        out = lexical
    elif not np.any(lexical):
        out = w2v_unit
    else:
        out = _l2_normalize(_MATCH_W2V_BLEND * w2v_unit + (1.0 - _MATCH_W2V_BLEND) * lexical)

    _MATCH_EMBED_CACHE[key] = out
    return out


def apply_exact_label_score(sim: torch.Tensor,
                            query_labels: list[str],
                            scene_labels: list[str],
                            exact_label_score: float = -1e9) -> torch.Tensor:
    """Force canonical exact-label pairs to at least exact_label_score."""
    if sim.numel() == 0:
        return sim
    if exact_label_score <= -1e9:
        return sim

    qcanon = [canonical_label(l) for l in query_labels]
    scanon = [canonical_label(l) for l in scene_labels]
    if not qcanon or not scanon:
        return sim

    match_mask = torch.zeros((len(qcanon), len(scanon)),
                             dtype=torch.bool,
                             device=sim.device)
    for qi, ql in enumerate(qcanon):
        if not ql:
            continue
        for si, sl in enumerate(scanon):
            if ql == sl:
                match_mask[qi, si] = True

    if not torch.any(match_mask):
        return sim
    score_tensor = torch.full_like(sim, float(exact_label_score))
    return torch.where(match_mask, torch.maximum(sim, score_tensor), sim)


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
#  Ray-visibility test (single ray)                                           ═
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
#  Colour helpers                                                             ═
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
    cmap = plt.get_cmap("viridis")
    return cmap(vals)[:, :3]


# ════════════════════════════════════════════════════════════════════════════
#  New: angular utilities + FOV maximisation                                  ═
# ════════════════════════════════════════════════════════════════════════════
def dir_to_yaw_pitch(v: np.ndarray):
    """Return yaw (around +Z, from +X toward +Y) and pitch (up) in radians."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    yaw = math.atan2(y, x)
    pitch = math.atan2(z, math.hypot(x, y))
    return yaw, pitch


def best_fov_window(yaws: np.ndarray,
                    pitches: np.ndarray,
                    hfov: float, vfov: float):
    """
    Given arrays of angles (rad), find subset that maximises count inside an
    axis-aligned yaw×pitch window of size hfov×vfov. Handles yaw wrap-around.
    Returns (selected_indices, max_count).
    """
    n = len(yaws)
    if n == 0:
        return np.array([], dtype=int), 0

    order = np.argsort(yaws)
    yaw_sorted = yaws[order]
    pit_sorted = pitches[order]
    idx_sorted = order

    yaw_ext = np.concatenate([yaw_sorted, yaw_sorted + 2 * math.pi])
    pit_ext = np.concatenate([pit_sorted, pit_sorted])
    idx_ext = np.concatenate([idx_sorted, idx_sorted])

    best_cnt = 0
    best_sel = np.array([], dtype=int)

    j = 0
    for s in range(n):
        y0 = yaw_ext[s]
        y1 = y0 + hfov
        j = max(j, s)
        while j < s + n and yaw_ext[j] <= y1 + 1e-9:
            j += 1

        if j <= s:
            continue

        # candidates within horizontal window
        cand_slice = slice(s, j)
        ps = pit_ext[cand_slice]
        ids = idx_ext[cand_slice]

        # 1D sliding window on pitch
        p_order = np.argsort(ps)
        ps = ps[p_order]
        ids = ids[p_order]

        t_end = 0
        for t_start in range(len(ps)):
            p0 = ps[t_start]
            p1 = p0 + vfov
            while t_end < len(ps) and ps[t_end] <= p1 + 1e-9:
                t_end += 1
            cnt = t_end - t_start
            if cnt > best_cnt:
                best_cnt = cnt
                best_sel = np.unique(ids[t_start:t_end])

        # (implicit: move to next s)

    return best_sel, int(best_cnt)


def average_direction(unit_dirs: np.ndarray, sel: np.ndarray):
    """Vector-average the selected unit directions; return unit 3D vector or None."""
    if sel.size == 0:
        return None
    m = unit_dirs[sel].mean(axis=0)
    n = np.linalg.norm(m)
    if n < 1e-8:
        return None
    return m / n


# ════════════════════════════════════════════════════════════════════════════
#  Main programme                                                             ═
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Compute and visualise localisation probability surface "
                    "for ScanScribe captions, plus FOV-weighted arrow field.")
    parser.add_argument("--root", required=True,
                        help="Parent folder of 3RScan/<scan_id>/")
    parser.add_argument("--graphs", required=True,
                        help="processed_data/{3dssg,scanscribe}/")
    parser.add_argument("--top_k", type=int, default=25,
                        help="How many object matches to keep per caption")
    parser.add_argument("--dynamic_top_k", action="store_true",
                        help="Use number of detected query objects as Top-K.")
    parser.add_argument("--score_threshold", type=float, default=-1.0,
                        help="Minimum cosine match score to keep (e.g. 0.1).")
    parser.add_argument("--grid_step", type=float, default=0.25,
                        help="XY grid spacing in metres")
    parser.add_argument("--query_limit", type=int,
                        help="Process only the first N captions (debug)")
    parser.add_argument("--show_heatmap", action="store_true",
                        help="Show 2-D Matplotlib heat-map")
    parser.add_argument("--show_3d", action="store_true",
                        help="Open Open3D viewer with mesh + probability spheres")

    # New: arrows / FOV options
    parser.add_argument("--show_arrows", action="store_true",
                        help="Show FOV-weighted arrow (quiver) plot")
    parser.add_argument("--h_fov_deg", type=float, default=100.0,
                        help="Horizontal FOV in degrees")
    parser.add_argument("--v_fov_deg", type=float, default=60.0,
                        help="Vertical FOV in degrees")
    parser.add_argument("--arrow_stride", type=int, default=2,
                        help="Plot every Nth grid camera (reduce clutter)")
    parser.add_argument("--arrow_len", type=float, default=0.0,
                        help="Max arrow length in metres (0 → 0.9*grid_step)")

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

        obj_ids = topk_matched_objects(qg, sg,
                                       k=args.top_k,
                                       score_threshold=args.score_threshold,
                                       dynamic_k=args.dynamic_top_k)
        if not obj_ids:
            print(f"[{qi}] {sid} : no cosine matches — skipped")
            continue

        # mesh + ray-caster
        mesh, tri2obj, obj2faces = load_scene(Path(args.root) / sid)
        rc = o3d.t.geometry.RaycastingScene()
        rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        verts = np.asarray(mesh.vertices)
        cams  = sample_grid(verts, step=args.grid_step)
        # Recompute gx/gy to know grid dimensions (match sample_grid logic)
        xs, ys = verts[:, 0], verts[:, 1]
        gx = np.arange(xs.min(), xs.max() + 1e-4, args.grid_step)
        gy = np.arange(ys.min(), ys.max() + 1e-4, args.grid_step)
        Nx, Ny = len(gx), len(gy)

        print(f"[{qi}] {sid}: grid {len(cams):,} pts  |  {len(obj_ids)} objs")

        # object centroids
        tris = np.asarray(mesh.triangles)
        centroids = {}
        for oid in obj_ids:
            faces = obj2faces.get(oid)
            if faces is not None and len(faces):
                centroids[oid] = verts[np.unique(tris[faces].ravel())].mean(0)

        if not centroids:
            print("    no centroids for matched objects — skipped\n")
            continue

        # ---- visibility + per-cam direction list (reuses single ray cast)
        visible_dirs = [[] for _ in range(len(cams))]
        for idx, cam in enumerate(cams):
            for oid, cen in centroids.items():
                if first_hit_is_object(cam, cen, oid, rc, tri2obj):
                    d = cen - cam
                    l = np.linalg.norm(d)
                    if l > 1e-6:
                        visible_dirs[idx].append(d / l)

        scores = np.array([len(v) for v in visible_dirs], dtype=np.int32)
        if scores.sum() == 0:
            print("    none of the matched objects visible — skipped\n")
            continue
        probs = scores / scores.sum()

        # --- 2-D heat-map
        if args.show_heatmap:
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(cams[:, 0], cams[:, 1], c=probs,
                             cmap="viridis", s=12)
            plt.colorbar(sc, label="probability")
            plt.title(f"{sid}  –  grid {args.grid_step} m")
            plt.axis("equal")
            plt.tight_layout()
            plt.show()

        # --- New: weighted arrow plot (quiver) -----------------------------
        if args.show_arrows:
            hfov = math.radians(args.h_fov_deg)
            vfov = math.radians(args.v_fov_deg)
            max_len = (0.9 * args.grid_step) if args.arrow_len <= 0 else args.arrow_len

            # Prepare arrays for quiver
            Qx, Qy, U, V, W = [], [], [], [], []  # positions, vectors, weights
            max_score = 0

            # Iterate subset of grid in a structured way using Nx,Ny
            stride = max(1, int(args.arrow_stride))
            for gy_i in range(0, Ny, stride):
                for gx_i in range(0, Nx, stride):
                    idx = gy_i * Nx + gx_i
                    dirs = np.asarray(visible_dirs[idx], dtype=np.float32)
                    if dirs.size == 0:
                        continue

                    # Yaw/pitch arrays
                    yaws = np.empty(len(dirs), dtype=np.float32)
                    pits = np.empty(len(dirs), dtype=np.float32)
                    for i, v in enumerate(dirs):
                        y, p = dir_to_yaw_pitch(v)
                        yaws[i] = y
                        pits[i] = p

                    sel, count = best_fov_window(yaws, pits, hfov, vfov)
                    if count == 0:
                        continue

                    mdir = average_direction(dirs, sel)
                    if mdir is None:
                        continue

                    # Project to XY for arrow direction
                    xy = mdir[:2]
                    nxy = np.linalg.norm(xy)
                    if nxy < 1e-8:
                        continue
                    xy_unit = xy / nxy

                    max_score = max(max_score, count)
                    Qx.append(cams[idx, 0])
                    Qy.append(cams[idx, 1])
                    U.append(float(xy_unit[0]))   # scaled later
                    V.append(float(xy_unit[1]))
                    W.append(count)

            if len(W) == 0:
                print("    arrows: no valid FOV windows — nothing to plot")
            else:
                W = np.array(W, dtype=np.float32)
                scale_fac = np.where(W > 0, W / W.max(), 0.0)
                U = np.array(U) * max_len * scale_fac
                V = np.array(V) * max_len * scale_fac

                plt.figure(figsize=(7, 7))
                plt.quiver(Qx, Qy, U, V, W, angles="xy", scale_units="xy",
                           scale=1.0, cmap="viridis", width=0.004,
                           minlength=0.01)
                plt.colorbar(label="max visible objects within FOV")
                plt.title(f"{sid} – FOV-weighted average directions "
                          f"(H={args.h_fov_deg}°, V={args.v_fov_deg}°, stride={stride})")
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
