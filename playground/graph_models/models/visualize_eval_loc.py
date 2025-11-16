#!/usr/bin/env python3
"""
visualize_eval_loc.py
---------------------
Evaluate localisation quality for ScanScribe style captions that come with
ground-truth camera poses. For every 3RScan scene this script:

1.  Loads the 3D-SSG scene graph and coloured instance mesh.
2.  Builds a caption SceneGraph from frame-level JSON metadata.
3.  Matches caption nodes to 3D objects via cosine similarity (Top-K).
4.  Samples a dense XY grid at eye height and casts visibility rays to the
    centroids of the matched objects.
5.  Converts first-hit counts into posterior probabilities.
6.  Extracts the ground-truth camera pose from the frame JSON.
7.  Visualises the probability heat-map, optional arrow field, and an Open3D
    scene with matched objects highlighted plus predicted/ground-truth cameras.
8.  Reports evaluation metrics such as NLL at the ground-truth, Hit@r mass, and
    Euclidean error between the MAP prediction and ground-truth camera.

The script reuses helper functions from visualize_loc_prob.py and constructs
caption graphs directly from the structured per-frame JSON to avoid any LLM
dependency during evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch

# --------------------------------------------------------------------------- #
# Repository imports                                                         #
# --------------------------------------------------------------------------- #

sys.path.append("../data_processing")
sys.path.append("../../../")

from scene_graph import SceneGraph  # noqa: E402
from create_text_embeddings import create_embedding_nlp  # noqa: E402
from visualize_3rscan_segments import build_segmented_mesh  # noqa: E402

# --------------------------------------------------------------------------- #
# Import helpers from visualize_loc_prob.py (hyphenated filename)            #
# --------------------------------------------------------------------------- #

_THIS_DIR = Path(__file__).resolve().parent
_VIZ = _THIS_DIR / "visualize_loc_prob.py"
if not _VIZ.exists():
    raise FileNotFoundError(f"Expected visualize_loc_prob.py at {_VIZ}")

from importlib.machinery import SourceFileLoader

vizmod = SourceFileLoader("vizlocprob_eval", str(_VIZ)).load_module()  # type: ignore

# Reuse helper functions
load_scene = vizmod.load_scene
topk_matched_objects = vizmod.topk_matched_objects
sample_grid = vizmod.sample_grid
first_hit_is_object = vizmod.first_hit_is_object
colour_objects = vizmod.colour_objects
colormap = vizmod.colormap
dir_to_yaw_pitch = getattr(vizmod, "dir_to_yaw_pitch", None)
best_fov_window = getattr(vizmod, "best_fov_window", None)
average_direction = getattr(vizmod, "average_direction", None)


# --------------------------------------------------------------------------- #
# Data containers                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class FrameSelection:
    frame: dict
    path: Path


@dataclass
class SceneMetrics:
    scene_id: str
    frame_id: str
    gt_prob: float
    nll: float
    hit_mass: float
    distance_error: float
    grid_points: int
    matched_objects: int


# --------------------------------------------------------------------------- #
# Caption graph construction utilities                                       #
# --------------------------------------------------------------------------- #

_EMBED_CACHE: Dict[str, np.ndarray] = {}


def _embed_word2vec(text: str) -> List[float]:
    key = text.strip().lower()
    cached = _EMBED_CACHE.get(key)
    if cached is None:
        vec = np.asarray(create_embedding_nlp(text), dtype=np.float32)
        cached = vec
        _EMBED_CACHE[key] = cached
    return cached.tolist()


def load_frame_jsons(desc_dir: Path) -> List[FrameSelection]:
    frames: List[FrameSelection] = []
    if not desc_dir.exists():
        return frames
    for path in sorted(desc_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue

        if isinstance(data, dict):
            frames.append(FrameSelection(frame=data, path=path))
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue
                virtual_name = path.with_name(f"{path.stem}_{idx:03d}{path.suffix}")
                frames.append(FrameSelection(frame=item, path=virtual_name))
    return frames


def select_frame(frames: List[FrameSelection],
                 policy: str,
                 frame_index: int,
                 rng: np.random.Generator) -> Optional[FrameSelection]:
    if not frames:
        return None

    if policy == "first":
        return frames[0]
    if policy == "index":
        return frames[frame_index % len(frames)]
    if policy == "random":
        return frames[int(rng.integers(0, len(frames)))]
    if policy == "max_visible":
        return max(frames,
                   key=lambda fs: len(fs.frame.get("visible_objects", {})))
    if policy == "max_pixels":
        def total_pixels(fs: FrameSelection) -> int:
            objs = fs.frame.get("visible_objects", {})
            return sum(int(obj.get("pixel_count", 0)) for obj in objs.values())

        return max(frames, key=total_pixels)

    raise ValueError(f"Unknown frame selection policy '{policy}'")


def frame_to_scenegraph(frame: dict,
                        embedding_type: str = "word2vec") -> Tuple[SceneGraph, Dict[int, dict]]:
    if embedding_type != "word2vec":
        raise ValueError("Only word2vec embedding supported for evaluation graphs.")

    visible_objects = frame.get("visible_objects", {}) or {}
    # Sort by descending pixel count to favour dominant objects for duplicate labels.
    sorted_items = sorted(
        visible_objects.items(),
        key=lambda kv: int(kv[1].get("pixel_count", 0)),
        reverse=True,
    )

    nodes: List[dict] = []
    label_lookup: Dict[str, List[int]] = {}
    meta: Dict[int, dict] = {}

    for new_id, (raw_id, obj) in enumerate(sorted_items):
        label = obj.get("label", f"object_{raw_id}")
        label_key = label.strip().lower()
        nodes.append({
            "id": new_id,
            "label": label,
            "attributes": [],
            "label_word2vec": _embed_word2vec(label),
            "attributes_word2vec": {"all": []},
        })
        label_lookup.setdefault(label_key, []).append(new_id)
        meta[new_id] = {
            "source_object_id": raw_id,
            "label": label,
            "centroid_world": np.asarray(obj.get("centroid_world", [0, 0, 0]),
                                         dtype=np.float32),
        }

    edges: List[dict] = []
    for rel in frame.get("spatial_relations", []) or []:
        subj = str(rel.get("subject", "")).strip().lower()
        obj = str(rel.get("object", "")).strip().lower()
        rel_type = rel.get("relation", "").strip()
        if not subj or not obj or not rel_type:
            continue
        subj_ids = label_lookup.get(subj)
        obj_ids = label_lookup.get(obj)
        if not subj_ids or not obj_ids:
            continue
        edges.append({
            "source": subj_ids[0],
            "target": obj_ids[0],
            "relationship": rel_type,
            "relation_word2vec": _embed_word2vec(rel_type),
        })

    graph_dict = {"nodes": nodes, "edges": edges}
    sg = SceneGraph(scene_id=frame.get("scene_index", "unknown_scene"),
                    txt_id=frame.get("image_index"),
                    graph_type="scanscribe",
                    graph=graph_dict,
                    embedding_type=embedding_type,
                    use_attributes=True)
    return sg, meta


# --------------------------------------------------------------------------- #
# Camera pose + metric helpers                                               #
# --------------------------------------------------------------------------- #

def camera_center_from_pose(pose: Iterable[Iterable[float]]) -> np.ndarray:
    mat = np.asarray(pose, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 scene_pose, got shape {mat.shape}")
    t = mat[:3, 3]
    return t.astype(np.float32)


def compute_metrics(cams: np.ndarray,
                    probs: np.ndarray,
                    gt_cam: np.ndarray,
                    eps: float,
                    hit_radius: float) -> Tuple[int, SceneMetrics]:
    pred_idx = int(np.argmax(probs))
    pred_cam = cams[pred_idx]

    distances = np.linalg.norm(cams - gt_cam[None, :], axis=1)
    gt_idx = int(np.argmin(distances))
    gt_prob = float(probs[gt_idx])
    gt_prob_clamped = max(gt_prob, eps)
    nll = float(-math.log(gt_prob_clamped))

    hit_mass = float(probs[distances <= hit_radius].sum())
    dist_err = float(np.linalg.norm(pred_cam - gt_cam))

    return pred_idx, SceneMetrics(
        scene_id="",
        frame_id="",
        gt_prob=gt_prob,
        nll=nll,
        hit_mass=hit_mass,
        distance_error=dist_err,
        grid_points=len(cams),
        matched_objects=0,
    )


def add_heatmap_markers(gt_cam: np.ndarray,
                        pred_cam: np.ndarray,
                        label_gt: str = "GT",
                        label_pred: str = "Pred") -> None:
    plt.scatter(gt_cam[0], gt_cam[1],
                c="red", marker="*", s=160,
                linewidths=1.2, edgecolors="black",
                label=label_gt)
    plt.scatter(pred_cam[0], pred_cam[1],
                c="orange", marker="o", s=80,
                linewidths=1.0, edgecolors="black",
                label=label_pred)
    plt.legend(loc="best")


def add_arrow_markers(gt_cam: np.ndarray,
                      pred_cam: np.ndarray) -> None:
    plt.scatter([gt_cam[0]], [gt_cam[1]],
                c="red", marker="*", s=160,
                linewidths=1.0, edgecolors="black",
                label="GT")
    plt.scatter([pred_cam[0]], [pred_cam[1]],
                c="orange", marker="o", s=80,
                linewidths=1.0, edgecolors="black",
                label="Pred")
    plt.legend(loc="best")


def create_camera_frustum(center: np.ndarray,
                          forward: Optional[np.ndarray],
                          colour: Tuple[float, float, float],
                          h_fov: float,
                          v_fov: float,
                          scale: float = 0.6) -> Optional[o3d.geometry.LineSet]:
    """Return a simple line-based frustum; None if forward dir unavailable."""
    if forward is None:
        return None
    fwd = np.asarray(forward, dtype=np.float64)
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        return None
    fwd /= norm

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(fwd, up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:
        return None
    right /= r_norm
    up = np.cross(right, fwd)

    depth = scale
    half_w = math.tan(h_fov / 2.0) * depth
    half_h = math.tan(v_fov / 2.0) * depth

    centre = np.asarray(center, dtype=np.float64)
    apex = centre
    base = centre + fwd * depth

    corners = [
        base + right * half_w + up * half_h,
        base - right * half_w + up * half_h,
        base - right * half_w - up * half_h,
        base + right * half_w - up * half_h,
    ]

    points = np.vstack([apex, *corners])
    lines = np.array([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ], dtype=np.int32)

    frustum = o3d.geometry.LineSet()
    frustum.points = o3d.utility.Vector3dVector(points)
    frustum.lines = o3d.utility.Vector2iVector(lines)
    colours = np.tile(np.asarray(colour, dtype=np.float64), (lines.shape[0], 1))
    frustum.colors = o3d.utility.Vector3dVector(colours)
    return frustum


# --------------------------------------------------------------------------- #
# Main evaluation pipeline                                                   #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate localisation probabilities against ground-truth camera poses."
    )
    parser.add_argument("--root", required=True,
                        help="Root directory containing <scene_id>/ meshes.")
    parser.add_argument("--graphs", required=True, type=Path,
                        help="processed_data directory holding 3dssg/*.pt files.")
    parser.add_argument("--query_root", type=Path,
                        help="Root containing per-scene output/descriptions/frame-*.json")

    parser.add_argument("--scene_ids", nargs="+",
                        help="Subset of scene IDs to evaluate. Defaults to intersection of graphs and query_root.")
    parser.add_argument("--max_scenes", type=int,
                        help="Limit number of scenes processed (after filtering).")

    parser.add_argument("--frame_policy",
                        choices=["first", "index", "random", "max_visible", "max_pixels"],
                        default="max_visible",
                        help="Strategy to pick which frame JSON to evaluate per scene.")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Frame index used when --frame_policy=index.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for random frame selection.")

    parser.add_argument("--top_k", type=int, default=25,
                        help="How many object matches to keep per caption.")
    parser.add_argument("--grid_step", type=float, default=0.25,
                        help="XY grid spacing in metres.")
    parser.add_argument("--eye_height", type=float, default=1.6,
                        help="Eye-height offset used by the grid sampler.")
    parser.add_argument("--prob_eps", type=float, default=1e-6,
                        help="Numerical epsilon when computing log-probabilities.")
    parser.add_argument("--hit_radius", type=float, default=0.5,
                        help="Radius (metres) used for Hit@r mass around ground-truth.")

    parser.add_argument("--show_heatmap", action="store_true",
                        help="Show 2-D probability scatter heatmap.")
    parser.add_argument("--show_3d", action="store_true",
                        help="Visualise mesh with probability spheres in Open3D.")
    parser.add_argument("--show_arrows", action="store_true",
                        help="Show FOV-weighted arrow (quiver) plot.")
    parser.add_argument("--h_fov_deg", type=float, default=100.0,
                        help="Horizontal FOV (degrees) for arrow aggregation.")
    parser.add_argument("--v_fov_deg", type=float, default=60.0,
                        help="Vertical FOV (degrees) for arrow aggregation.")
    parser.add_argument("--arrow_stride", type=int, default=2,
                        help="Plot every Nth grid camera in the arrow field.")
    parser.add_argument("--arrow_len", type=float, default=0.0,
                        help="Maximum arrow length (metres). 0 → 0.9 * grid_step.")

    parser.add_argument("--save_metrics", type=Path,
                        help="Optional path to save per-scene metrics as JSON.")
    return parser.parse_args()


def load_scene_graphs(graphs_dir: Path) -> Dict[str, SceneGraph]:
    g3d_path = graphs_dir / "3dssg" / "3dssg_graphs_processed_edgelists_relationembed.pt"
    if not g3d_path.exists():
        raise FileNotFoundError(g3d_path)
    g3d = torch.load(g3d_path, map_location="cpu")
    scenes: Dict[str, SceneGraph] = {}
    for sid, graph in g3d.items():
        scenes[sid] = SceneGraph(sid,
                                 graph_type="3dssg",
                                 graph=graph,
                                 max_dist=1.0,
                                 embedding_type="word2vec",
                                 use_attributes=True)
    return scenes


def ensure_query_root(query_root: Optional[Path], root: Path) -> Path:
    if query_root is not None:
        return query_root
    return root


def evaluate_scene(scene_id: str,
                   scene_graph: SceneGraph,
                   args: argparse.Namespace,
                   rng: np.random.Generator) -> Optional[SceneMetrics]:
    mesh_root = Path(args.root)
    scene_dir = mesh_root / scene_id
    if not scene_dir.exists():
        print(f"[WARN] Scene directory missing for {scene_id} — skipped.")
        return None

    query_root = ensure_query_root(args.query_root, Path(args.root))
    desc_dir = query_root / scene_id / "output" / "descriptions"
    if not desc_dir.exists():
        # Fallback: allow descriptions alongside mesh root (already same path)
        desc_dir = scene_dir / "output" / "descriptions"
    frames = load_frame_jsons(desc_dir)
    if not frames:
        print(f"[WARN] No frame JSONs under {desc_dir} — skipped.")
        return None

    selection = select_frame(frames, args.frame_policy, args.frame_index, rng)
    if selection is None:
        print(f"[WARN] Frame selection failed for {scene_id} — skipped.")
        return None

    frame = selection.frame
    try:
        caption_graph, _ = frame_to_scenegraph(frame)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to build caption graph for {scene_id}: {exc}")
        return None

    gt_pose = frame.get("scene_pose")
    if gt_pose is None:
        print(f"[WARN] scene_pose missing in {selection.path} — skipped.")
        return None

    pose_mat = np.asarray(gt_pose, dtype=np.float64)
    gt_cam = camera_center_from_pose(pose_mat)
    rot_cam_world = pose_mat[:3, :3]
    forward_cv = rot_cam_world @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_o3d = forward_cv
    norm_forward = np.linalg.norm(forward_o3d)
    gt_dir = forward_o3d / norm_forward if norm_forward > 1e-6 else None

    obj_ids = topk_matched_objects(caption_graph, scene_graph, k=args.top_k)
    if not obj_ids:
        print(f"[WARN] {scene_id}: no cosine matches — skipped.")
        return None

    mesh, tri2obj, obj2faces = load_scene(scene_dir)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    verts = np.asarray(mesh.vertices)
    cams = sample_grid(verts, step=args.grid_step, z_eye=args.eye_height)

    xs, ys = verts[:, 0], verts[:, 1]
    gx = np.arange(xs.min(), xs.max() + 1e-4, args.grid_step)
    gy = np.arange(ys.min(), ys.max() + 1e-4, args.grid_step)
    Nx, Ny = len(gx), len(gy)

    tris = np.asarray(mesh.triangles)
    centroids: Dict[int, np.ndarray] = {}
    for oid in obj_ids:
        faces = obj2faces.get(int(oid))
        if faces is None or not len(faces):
            continue
        centroids[int(oid)] = verts[np.unique(tris[faces].ravel())].mean(axis=0)

    if not centroids:
        print(f"[WARN] {scene_id}: matched objects missing geometry — skipped.")
        return None

    visible_dirs: List[List[np.ndarray]] = [[] for _ in range(len(cams))]
    for idx, cam in enumerate(cams):
        for oid, centre in centroids.items():
            if first_hit_is_object(cam, centre, oid, rc, tri2obj):
                d = centre - cam
                l = np.linalg.norm(d)
                if l > 1e-6:
                    visible_dirs[idx].append(d / l)

    counts = np.array([len(v) for v in visible_dirs], dtype=np.int32)
    total = counts.sum()
    if total == 0:
        print(f"[WARN] {scene_id}: matched objects invisible from grid — skipped.")
        return None
    probs = counts / total

    pred_idx, metrics = compute_metrics(cams, probs, gt_cam,
                                        eps=args.prob_eps,
                                        hit_radius=args.hit_radius)
    metrics.scene_id = scene_id
    metrics.frame_id = str(frame.get("image_index", selection.path.name))
    metrics.matched_objects = len(obj_ids)

    pred_cam_prob = cams[pred_idx]

    # ---- Arrow-based aggregation (computed regardless of plotting)
    arrow_positions: List[np.ndarray] = []
    arrow_dirs: List[np.ndarray] = []
    arrow_weights: List[float] = []

    have_arrow_helpers = bool(dir_to_yaw_pitch and best_fov_window and average_direction)
    if have_arrow_helpers:
        hfov = math.radians(args.h_fov_deg)
        vfov = math.radians(args.v_fov_deg)
        stride = max(1, int(args.arrow_stride))
        for gy_i in range(0, Ny, stride):
            for gx_i in range(0, Nx, stride):
                idx = gy_i * Nx + gx_i
                dirs = np.asarray(visible_dirs[idx], dtype=np.float32)
                if dirs.size == 0:
                    continue
                yaws = np.empty(len(dirs), dtype=np.float32)
                pits = np.empty(len(dirs), dtype=np.float32)
                for i, vec in enumerate(dirs):
                    yaw, pit = dir_to_yaw_pitch(vec)  # type: ignore[arg-type]
                    yaws[i] = yaw
                    pits[i] = pit
                sel, count = best_fov_window(yaws, pits, hfov, vfov)  # type: ignore[arg-type]
                if count == 0:
                    continue
                mdir = average_direction(dirs, sel)  # type: ignore[arg-type]
                if mdir is None:
                    continue
                arrow_positions.append(cams[idx])
                arrow_dirs.append(mdir)
                arrow_weights.append(float(count))

    pred_cam = pred_cam_prob
    pred_dir: Optional[np.ndarray] = None
    pred_source = "grid_probability"
    if arrow_weights:
        w_arr = np.asarray(arrow_weights, dtype=np.float32)
        max_w = float(w_arr.max())
        winners = [i for i, w in enumerate(arrow_weights)
                   if math.isclose(w, max_w, rel_tol=1e-6, abs_tol=1e-6)]
        winner_pos = np.stack([arrow_positions[i] for i in winners], axis=0)
        pred_cam = winner_pos.mean(axis=0)

        winner_dirs = np.stack([arrow_dirs[i] for i in winners], axis=0)
        pred_dir = winner_dirs.mean(axis=0)
        norm_dir = float(np.linalg.norm(pred_dir))
        if norm_dir > 1e-6:
            pred_dir /= norm_dir
        else:
            pred_dir = None
        pred_source = "arrow_field"

    metrics.distance_error = float(np.linalg.norm(pred_cam - gt_cam))

    print(f"    predicted camera ({pred_source}): "
          f"{pred_cam.tolist()}")
    if pred_dir is not None:
        print(f"    approx. viewing direction: {pred_dir.tolist()} \n")
    else:
        print()

    if args.show_heatmap:
        plt.figure(figsize=(6.5, 6.2))
        sc = plt.scatter(cams[:, 0], cams[:, 1], c=probs,
                         cmap="viridis", s=14)
        plt.colorbar(sc, label="Probability")
        plt.axis("equal")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"{scene_id} · {metrics.frame_id} · grid {args.grid_step:.2f} m")
        add_heatmap_markers(gt_cam, pred_cam,
                            label_pred=f"Pred ({pred_source})")
        plt.tight_layout()
        plt.show()

    if args.show_arrows:
        if arrow_weights:
            hfov = math.radians(args.h_fov_deg)
            vfov = math.radians(args.v_fov_deg)
            max_len = (0.9 * args.grid_step) if args.arrow_len <= 0 else args.arrow_len
            W_np = np.asarray(arrow_weights, dtype=np.float32)
            scale = np.where(W_np > 0, W_np / W_np.max(), 0.0)
            dirs_xy = np.asarray([d[:2] for d in arrow_dirs], dtype=np.float32)
            norms = np.linalg.norm(dirs_xy, axis=1, keepdims=True)
            norms = np.where(norms < 1e-8, 1.0, norms)
            dirs_xy /= norms
            U_np = dirs_xy[:, 0] * max_len * scale
            V_np = dirs_xy[:, 1] * max_len * scale
            Qx = [float(p[0]) for p in arrow_positions]
            Qy = [float(p[1]) for p in arrow_positions]

            plt.figure(figsize=(7, 6.5))
            plt.quiver(Qx, Qy, U_np, V_np, W_np,
                       angles="xy", scale_units="xy", scale=1.0,
                       cmap="viridis", width=0.004, minlength=0.01)
            plt.colorbar(label="Max visible objects within FOV")
            plt.axis("equal")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.title(f"{scene_id} · {metrics.frame_id} · FOV arrows "
                      f"(H={math.degrees(hfov):.0f}°, V={math.degrees(vfov):.0f}°)")
            add_arrow_markers(gt_cam, pred_cam)
            plt.tight_layout()
            plt.show()
        else:
            print("    [info] Arrow plot skipped (no valid FOV windows).")

    if args.show_3d:
        matched_set: set[int] = {int(o) for o in obj_ids}
        frustum_scale = max(args.grid_step * 3.0, 0.6)
        try:
            mesh_vis, obj_stats = build_segmented_mesh(scene_dir, seed=42)
            colours = np.asarray(mesh_vis.vertex_colors)
            highlight = np.array([1.0, 0.3, 0.3], dtype=np.float64)
            for stats in obj_stats:
                oid = int(stats["object_id"])
                if oid in matched_set:
                    idx = stats.get("vertex_indices")
                    if idx is not None:
                        colours[idx] = np.clip(0.55 * colours[idx] + 0.45 * highlight, 0.0, 1.0)
            mesh_vis.vertex_colors = o3d.utility.Vector3dVector(colours)
            if not mesh_vis.has_vertex_normals():
                mesh_vis.compute_vertex_normals()
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] Segment mesh loading failed ({exc}) — falling back to legacy mesh.")
            mesh_vis = colour_objects(mesh, obj2faces, obj_ids)
            obj_stats = []
        if not mesh_vis.has_vertex_normals():
            mesh_vis.compute_vertex_normals()

        from open3d.visualization import gui, rendering

        global GUI_INITIALISED
        if not GUI_INITIALISED:
            gui.Application.instance.initialize()
            GUI_INITIALISED = True

        vis = o3d.visualization.O3DVisualizer(f"{scene_id} – localisation eval", 1280, 800)
        vis.show_settings = False

        material = rendering.MaterialRecord()
        material.shader = "defaultLit"
        vis.add_geometry("mesh", mesh_vis, material)

        text_added = set()
        if obj_stats:
            bbox_material = rendering.MaterialRecord()
            bbox_material.shader = "unlitLine"
            bbox_material.line_width = 1.5
            for stats in obj_stats:
                oid = int(stats["object_id"])
                label = stats.get("label") or f"id_{oid}"
                centroid = np.asarray(stats["centroid"]) if "centroid" in stats else None
                if centroid is not None and tuple(centroid) not in text_added:
                    vis.add_3d_label(centroid, f"{oid}: {label}")
                    text_added.add(tuple(centroid))
                if oid in matched_set and "bbox" in stats:
                    vis.add_geometry(f"bbox_{oid}", stats["bbox"], bbox_material)

        # Probability spheres
        prob_material = rendering.MaterialRecord()
        prob_material.shader = "defaultLit"
        prob_material.base_color = [1.0, 1.0, 1.0, 1.0]
        for idx_point, (point, colour) in enumerate(zip(cams, colormap(probs))):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            s.translate(point)
            s.paint_uniform_color(colour)
            if not s.has_vertex_normals():
                s.compute_vertex_normals()
            vis.add_geometry(f"prob_{idx_point}", s, prob_material)

        gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        gt_sphere.translate(gt_cam)
        gt_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        if not gt_sphere.has_vertex_normals():
            gt_sphere.compute_vertex_normals()
        vis.add_geometry("gt_cam", gt_sphere, material)
        vis.add_3d_label(gt_cam, "GT")

        pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.085)
        pred_sphere.translate(pred_cam)
        pred_sphere.paint_uniform_color([1.0, 0.55, 0.0])
        if not pred_sphere.has_vertex_normals():
            pred_sphere.compute_vertex_normals()
        vis.add_geometry("pred_cam", pred_sphere, material)
        vis.add_3d_label(pred_cam, f"Pred ({pred_source})")

        frustum_gt = create_camera_frustum(gt_cam, gt_dir,
                                           colour=(1.0, 0.0, 0.0),
                                           h_fov=math.radians(args.h_fov_deg),
                                           v_fov=math.radians(args.v_fov_deg),
                                           scale=frustum_scale)
        frustum_pred = create_camera_frustum(pred_cam, pred_dir,
                                             colour=(1.0, 0.6, 0.0),
                                             h_fov=math.radians(args.h_fov_deg),
                                             v_fov=math.radians(args.v_fov_deg),
                                             scale=frustum_scale)
        if frustum_gt is not None:
            frustum_mat = rendering.MaterialRecord()
            frustum_mat.shader = "unlitLine"
            frustum_mat.line_width = 2.0
            vis.add_geometry("frustum_gt", frustum_gt, frustum_mat)
        if frustum_pred is not None:
            frustum_mat_pred = rendering.MaterialRecord()
            frustum_mat_pred.shader = "unlitLine"
            frustum_mat_pred.line_width = 2.0
            vis.add_geometry("frustum_pred", frustum_pred, frustum_mat_pred)

        vis.reset_camera_to_default()
        gui.Application.instance.add_window(vis)
        gui.Application.instance.run()

    return metrics


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)

    scenes = load_scene_graphs(args.graphs)

    candidate_ids = list(scenes.keys())
    if args.scene_ids:
        scene_set = set(args.scene_ids)
        candidate_ids = [sid for sid in candidate_ids if sid in scene_set]
    else:
        # Filter by available descriptions
        query_root = ensure_query_root(args.query_root, Path(args.root))
        candidate_ids = [
            sid for sid in candidate_ids
            if (query_root / sid / "output" / "descriptions").exists()
            or (Path(args.root) / sid / "output" / "descriptions").exists()
        ]

    candidate_ids.sort()
    if args.max_scenes is not None:
        candidate_ids = candidate_ids[: args.max_scenes]

    print(f"Evaluating {len(candidate_ids)} scene(s)...\n")

    metrics_list: List[SceneMetrics] = []
    for idx, sid in enumerate(candidate_ids, start=1):
        print(f"[{idx:03d}/{len(candidate_ids):03d}] {sid}")
        scene_metrics = evaluate_scene(sid, scenes[sid], args, rng)
        if scene_metrics is None:
            continue
        metrics_list.append(scene_metrics)
        print(f"    frame: {scene_metrics.frame_id}")
        print(f"    matches: {scene_metrics.matched_objects} | grid pts: {scene_metrics.grid_points}")
        print(f"    gt_prob: {scene_metrics.gt_prob:.4f} | nll: {scene_metrics.nll:.3f}")
        print(f"    hit@{args.hit_radius:.2f}m: {scene_metrics.hit_mass:.3f} | "
              f"dist_err: {scene_metrics.distance_error:.3f} m\n")

    if not metrics_list:
        print("No scenes produced metrics. Nothing to report.")
        return

    # Aggregate metrics
    def agg(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(np.median(arr))

    mean_gt, med_gt = agg([m.gt_prob for m in metrics_list])
    mean_nll, med_nll = agg([m.nll for m in metrics_list])
    mean_hit, med_hit = agg([m.hit_mass for m in metrics_list])
    mean_err, med_err = agg([m.distance_error for m in metrics_list])

    print("Aggregate metrics ---------------------------------------")
    print(f"  GT probability     : mean={mean_gt:.4f} | median={med_gt:.4f}")
    print(f"  NLL (surprisal)    : mean={mean_nll:.3f} | median={med_nll:.3f}")
    print(f"  Hit@{args.hit_radius:.2f}m       : mean={mean_hit:.3f} | median={med_hit:.3f}")
    print(f"  Distance error (m) : mean={mean_err:.3f} | median={med_err:.3f}")
    print("---------------------------------------------------------\n")

    if args.save_metrics:
        payload = [
            {
                "scene_id": m.scene_id,
                "frame_id": m.frame_id,
                "gt_prob": m.gt_prob,
                "nll": m.nll,
                "hit_mass": m.hit_mass,
                "distance_error": m.distance_error,
                "grid_points": m.grid_points,
                "matched_objects": m.matched_objects,
            }
            for m in metrics_list
        ]
        args.save_metrics.write_text(json.dumps({
            "metrics": payload,
            "aggregate": {
                "gt_prob": {"mean": mean_gt, "median": med_gt},
                "nll": {"mean": mean_nll, "median": med_nll},
                "hit_mass": {"mean": mean_hit, "median": med_hit},
                "distance_error": {"mean": mean_err, "median": med_err},
                "hit_radius": args.hit_radius,
                "top_k": args.top_k,
                "grid_step": args.grid_step,
            },
        }, indent=2))
        print(f"Metrics saved to {args.save_metrics}")


GUI_INITIALISED = False

if __name__ == "__main__":
    main()
