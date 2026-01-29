#!/usr/bin/env python3
"""
visualize_eval_loc.py
---------------------
Evaluate localisation quality for ScanScribe style captions that come with
ground-truth camera poses. For every 3RScan scene this script:

1.  Loads processed 3D-SSG graphs, per-scene meshes, and selects a frame JSON
    from `output/descriptions` according to the requested policy.
2.  Builds a caption SceneGraph from visible_objects + spatial_relations in the
    frame (word2vec embeddings only).
3.  Matches caption nodes to 3D objects via cosine similarity and keeps the top
    K candidates.
4.  Loads the coloured mesh, samples an XY grid at eye height, and computes
    centroids for each matched object.
5.  Casts rays from every grid camera to those centroids, counts first hits, and
    derives visibility probabilities.
6.  Extracts the ground-truth camera centre/direction from `scene_pose` and
    reports probability, NLL, Hit@r, and distance error at the ground-truth.
7.  Optionally aggregates viewing directions into a FOV-weighted arrow field
    and prints the strongest pose candidates.
8.  Chooses a final camera prediction (argmax/random/cluster-weighted) from the
    grid or arrow candidates, optionally averaging directions.
9.  Visualises heatmap scatter, arrow quiver, and an Open3D scene with matched
    objects, probability spheres, and GT/predicted cameras.
10. Logs a per-scene table, aggregate metrics, and optionally saves a JSON dump.

The script reuses helper functions from visualize_loc_prob.py and constructs
caption graphs directly from the structured per-frame JSON to avoid any LLM
dependency during evaluation.


### WITH IOU CALCULATION AND VISUALISATION
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
    iou_error: Optional[float] = None


def build_metrics_table(metrics_list: List[SceneMetrics], hit_radius: float) -> str:
    headers = [
        "Scene",
        "Frame",
        "GT Prob",
        "NLL",
        f"Hit@{hit_radius:.2f}m",
        "Err (m)",
        "IoU err",
        "Matches",
        "Grid pts",
    ]
    rows: List[List[str]] = []
    for m in metrics_list:
        rows.append([
            m.scene_id,
            m.frame_id,
            f"{m.gt_prob:.4f}",
            f"{m.nll:.3f}",
            f"{m.hit_mass:.3f}",
            f"{m.distance_error:.3f}",
            "-" if m.iou_error is None else f"{m.iou_error:.3f}",
            str(m.matched_objects),
            str(m.grid_points),
        ])

    if not rows:
        return ""

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def format_args_section(args: argparse.Namespace) -> str:
    """Return a human-readable list of CLI parameters."""

    def _stringify(value: object) -> str:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(_stringify(v) for v in value) + "]"
        if isinstance(value, dict):
            items = ", ".join(f"{k}: {_stringify(v)}" for k, v in value.items())
            return "{" + items + "}"
        return str(value)

    lines = ["Parameters used", "---------------"]
    for key in sorted(vars(args)):
        if key.startswith("_"):
            continue
        value = getattr(args, key)
        lines.append(f"{key}: {_stringify(value)}")
    return "\n".join(lines)


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


def _camera_axes_from_forward(forward: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return orthonormal (fwd, right, up) axes derived from a forward vector."""
    fwd = np.asarray(forward, dtype=np.float64)
    norm = float(np.linalg.norm(fwd))
    if norm < 1e-6:
        return None
    fwd /= norm

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(fwd, up)
    r_norm = float(np.linalg.norm(right))
    if r_norm < 1e-6:
        return None
    right /= r_norm
    up = np.cross(right, fwd)
    u_norm = float(np.linalg.norm(up))
    if u_norm < 1e-6:
        return None
    up /= u_norm
    return fwd, right, up


def _visible_triangles_from_view(cam: np.ndarray,
                                 forward: Optional[np.ndarray],
                                 hfov: float,
                                 vfov: float,
                                 rc: o3d.t.geometry.RaycastingScene,
                                 geom_id: int,
                                 tri_pts: np.ndarray,
                                 tri_centroids: np.ndarray,
                                 near: float = 0.05,
                                 far: Optional[float] = None) -> set[int]:
    """
    Return triangle indices visible from a camera: inside frustum (all verts),
    and first-hit occlusion check via centroid rays.
    """
    if forward is None:
        return set()
    axes = _camera_axes_from_forward(forward)
    if axes is None:
        return set()
    fwd_axis, right_axis, up_axis = axes

    cam = np.asarray(cam, dtype=np.float64)
    rel = tri_pts - cam[None, None, :]        # [T,3,3]

    fwd = rel @ fwd_axis
    right = rel @ right_axis
    up = rel @ up_axis

    near = max(float(near), 1e-4)
    in_front = np.all(fwd > near, axis=1)
    if far is not None:
        in_front &= np.all(fwd < float(far), axis=1)

    tan_h = math.tan(hfov * 0.5)
    tan_v = math.tan(vfov * 0.5)

    inside_h = np.all(np.abs(right) <= fwd * tan_h, axis=1)
    inside_v = np.all(np.abs(up) <= fwd * tan_v, axis=1)

    frustum_mask = in_front & inside_h & inside_v
    if not np.any(frustum_mask):
        return set()

    sel_idx = np.nonzero(frustum_mask)[0]

    # Occlusion test: cast a ray to the centroid
    vecs = tri_centroids[sel_idx] - cam[None, :]
    dist = np.linalg.norm(vecs, axis=1)
    valid = dist > 1e-6
    if not np.any(valid):
        return set()

    sel_idx = sel_idx[valid]
    dirs = vecs[valid] / dist[valid][:, None]

    rays = np.concatenate(
        [np.repeat(cam[None, :], len(sel_idx), axis=0), dirs],
        axis=1,
    ).astype(np.float32)

    res = rc.cast_rays(o3d.core.Tensor(rays))
    prim_ids = np.asarray(res["primitive_ids"].numpy())
    geom_ids = np.asarray(res["geometry_ids"].numpy())

    hit_mask = (prim_ids == sel_idx) & (geom_ids == geom_id)
    return {int(i) for i in sel_idx[hit_mask]}


def compute_view_iou_error(gt_cam: np.ndarray,
                           gt_dir: Optional[np.ndarray],
                           pred_cam: np.ndarray,
                           pred_dir: Optional[np.ndarray],
                           hfov: float,
                           vfov: float,
                           rc: o3d.t.geometry.RaycastingScene,
                           geom_id: int,
                           tri_pts: np.ndarray,
                           tri_centroids: np.ndarray,
                           tri_areas: np.ndarray,
                           near: float = 0.05,
                           far: Optional[float] = None) -> Tuple[Optional[float], Optional[float], set[int], set[int]]:
    """
    Return (IoU, IoU error, gt_set, pred_set) between GT and predicted views.
    IoU error = 1 - IoU. Returns (None, None, sets) if insufficient data.
    """
    if gt_dir is None or pred_dir is None:
        return None, None, set(), set()
    gt_vis = _visible_triangles_from_view(gt_cam, gt_dir, hfov, vfov,
                                          rc, geom_id, tri_pts, tri_centroids,
                                          near=near, far=far)
    pred_vis = _visible_triangles_from_view(pred_cam, pred_dir, hfov, vfov,
                                            rc, geom_id, tri_pts, tri_centroids,
                                            near=near, far=far)
    if not gt_vis and not pred_vis:
        return None, None, gt_vis, pred_vis

    inter = gt_vis & pred_vis
    union = gt_vis | pred_vis
    if not union:
        return None, None, gt_vis, pred_vis

    inter_area = float(tri_areas[list(inter)].sum()) if inter else 0.0
    union_area = float(tri_areas[list(union)].sum())
    if union_area <= 1e-9:
        return None, None, gt_vis, pred_vis
    iou = inter_area / union_area
    return iou, 1.0 - iou, gt_vis, pred_vis


def _cluster_weighted_prediction(positions: np.ndarray,
                                 weights: np.ndarray,
                                 bandwidth: float,
                                 max_points: int) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Return a weighted-average position that emphasises local clusters."""
    if len(positions) == 0:
        raise ValueError("No candidate positions available for prediction.")

    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)

    bandwidth = max(float(bandwidth), 1e-6)
    max_points = max(1, int(max_points))

    idx_sorted = np.argsort(weights)
    if len(idx_sorted) > max_points:
        idx_sorted = idx_sorted[-max_points:]

    subset_positions = positions[idx_sorted]
    subset_weights = weights[idx_sorted]
    subset_weights /= subset_weights.sum()

    if len(subset_positions) == 1:
        return subset_positions[0], [int(idx_sorted[0])], np.asarray([1.0], dtype=np.float64)

    diff = subset_positions[:, None, :] - subset_positions[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    kernel = np.exp(-dist2 / (2.0 * bandwidth * bandwidth))
    density = kernel @ subset_weights
    cluster_weights = subset_weights * density
    total = cluster_weights.sum()
    if total <= 0:
        cluster_weights = subset_weights
        total = cluster_weights.sum()
    cluster_weights /= total
    pred = np.sum(cluster_weights[:, None] * subset_positions, axis=0)

    return pred, [int(idx) for idx in idx_sorted], cluster_weights


def select_prediction_point(positions: np.ndarray,
                            weights: np.ndarray,
                            strategy: str,
                            rng: np.random.Generator,
                            bandwidth: float,
                            max_points: int) -> Tuple[np.ndarray, List[int], np.ndarray]:
    """Select a predicted camera position according to the requested strategy."""
    if len(positions) == 0:
        raise ValueError("No candidate positions available for prediction.")

    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)
    total = weights.sum()

    if strategy == "argmax" or len(positions) == 1:
        idx = int(np.argmax(weights))
        return positions[idx], [idx], np.asarray([1.0], dtype=np.float64)

    if strategy == "random":
        probs = weights / total
        idx = int(rng.choice(len(positions), p=probs))
        return positions[idx], [idx], np.asarray([1.0], dtype=np.float64)

    # Default: weighted cluster-aware prediction.
    return _cluster_weighted_prediction(positions,
                                        weights,
                                        bandwidth=bandwidth,
                                        max_points=max_points)


def top_n_fov_poses(positions: np.ndarray,
                    weights: np.ndarray,
                    n: int,
                    rng: np.random.Generator,
                    directions: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
    """Return up to n pose/direction pairs prioritising highest FOV-weighted probability."""
    if n <= 0:
        return []
    if len(positions) == 0 or len(weights) == 0:
        return []
    if len(positions) != len(weights):
        raise ValueError("Positions and weights must have the same length.")
    if directions is not None and len(directions) != len(positions):
        raise ValueError("Directions must align with positions.")

    weights = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
    if not np.any(weights > 0):
        weights = np.ones_like(weights)

    max_w = float(weights.max())
    top_idx = np.where(weights == max_w)[0]

    if len(top_idx) > n:
        chosen = rng.choice(top_idx, size=n, replace=False)
    else:
        sorted_idx = np.argsort(-weights)
        chosen = sorted_idx[: min(n, len(sorted_idx))]

    results: List[Dict[str, object]] = []
    for i in chosen:
        pose_xy = [float(positions[i][0]), float(positions[i][1])]
        if directions is not None:
            dir_vec = np.asarray(directions[i], dtype=np.float64)
            norm = float(np.linalg.norm(dir_vec))
            dir_out = (dir_vec / norm).tolist() if norm > 1e-6 else None
        else:
            dir_out = None
        results.append({
            "pose": pose_xy,
            "direction": dir_out,
        })
    return results


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
    parser.add_argument("--visualize_scene",
                        help="Scene ID to focus on for visualisation. Overrides --scene_ids when set.")

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
    parser.add_argument("--prediction_strategy",
                        choices=["argmax", "random", "weighted"],
                        default="weighted",
                        help="How to convert candidate positions into a final camera prediction.")
    parser.add_argument("--cluster_bandwidth", type=float, default=0.75,#0.75 #0.25
                        help="Bandwidth (metres) for the weighted cluster strategy.")
    parser.add_argument("--max_cluster_points", type=int, default=512,#512 # 20
                        help="Maximum candidates used when computing cluster-aware predictions.")

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
    parser.add_argument("--log_file", type=Path, default=Path("eval_loc_summary.log"),
                        help="Path to write a plain-text summary log.")
    parser.add_argument("--top_pose_count", type=int, default=5,
                        help="Number of top FOV-weighted poses to list.")
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
    mesh_id = rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    tri_pts = verts[tris]
    tri_vecs = tri_pts[:, 1] - tri_pts[:, 0]
    tri_vecs_b = tri_pts[:, 2] - tri_pts[:, 0]
    tri_cross = np.cross(tri_vecs, tri_vecs_b)
    tri_areas = 0.5 * np.linalg.norm(tri_cross, axis=1)
    tri_centroids = tri_pts.mean(axis=1)
    cams = sample_grid(verts, step=args.grid_step, z_eye=args.eye_height)

    xs, ys = verts[:, 0], verts[:, 1]
    gx = np.arange(xs.min(), xs.max() + 1e-4, args.grid_step)
    gy = np.arange(ys.min(), ys.max() + 1e-4, args.grid_step)
    Nx, Ny = len(gx), len(gy)

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

    hfov_rad = math.radians(args.h_fov_deg)
    vfov_rad = math.radians(args.v_fov_deg)

    # ---- Arrow-based aggregation (computed regardless of plotting)
    arrow_positions: List[np.ndarray] = []
    arrow_dirs: List[np.ndarray] = []
    arrow_weights: List[float] = []

    have_arrow_helpers = bool(dir_to_yaw_pitch and best_fov_window and average_direction)
    if have_arrow_helpers:
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
                sel, count = best_fov_window(yaws, pits, hfov_rad, vfov_rad)  # type: ignore[arg-type]
                if count == 0:
                    continue
                mdir = average_direction(dirs, sel)  # type: ignore[arg-type]
                if mdir is None:
                    continue
                arrow_positions.append(cams[idx])
                arrow_dirs.append(mdir)
                arrow_weights.append(float(count))

    candidate_dirs: Optional[np.ndarray] = None
    candidate_source = "grid_probability"
    if arrow_weights:
        candidate_source = "arrow_field"
        candidate_positions = np.asarray(arrow_positions, dtype=np.float64)
        candidate_weights = np.asarray(arrow_weights, dtype=np.float64)
        candidate_dirs = np.asarray(arrow_dirs, dtype=np.float64)
        top_fov_poses = top_n_fov_poses(candidate_positions,
                                        candidate_weights,
                                        n=args.top_pose_count,
                                        rng=rng,
                                        directions=candidate_dirs)
        print(f"    top-{args.top_pose_count} FOV-weighted poses (pose x,y + dir): "
              f"{top_fov_poses}")
    else:
        candidate_positions = cams
        candidate_weights = probs

    pred_dir: Optional[np.ndarray] = None
    try:
        pred_cam, selection_idx, selection_weights = select_prediction_point(
            candidate_positions,
            candidate_weights,
            strategy=args.prediction_strategy,
            rng=rng,
            bandwidth=args.cluster_bandwidth,
            max_points=args.max_cluster_points,
        )
    except ValueError:
        pred_cam = pred_cam_prob
        selection_idx = [int(pred_idx)]
        selection_weights = np.asarray([1.0], dtype=np.float64)

    if candidate_dirs is not None and selection_idx:
        dir_vectors = candidate_dirs[selection_idx]
        weight_vec = selection_weights
        if weight_vec.shape[0] != len(selection_idx):
            weight_vec = np.ones(len(selection_idx), dtype=np.float64)
        weight_vec = np.clip(weight_vec, 0.0, None)
        if not np.any(weight_vec > 0):
            weight_vec = np.ones_like(weight_vec)
        weight_vec /= weight_vec.sum()
        mean_dir = np.sum(weight_vec[:, None] * dir_vectors, axis=0)
        norm_dir = float(np.linalg.norm(mean_dir))
        if norm_dir > 1e-6:
            pred_dir = mean_dir / norm_dir

    pred_source = f"{candidate_source}:{args.prediction_strategy}"
    metrics.distance_error = float(np.linalg.norm(pred_cam - gt_cam))

    print(f"    predicted camera ({pred_source}): "
          f"{pred_cam.tolist()}")
    if pred_dir is not None:
        print(f"    approx. viewing direction: {pred_dir.tolist()}")
    else:
        print("    approx. viewing direction: n/a (no directional vote)")

    iou_val, iou_err, gt_vis_set, pred_vis_set = compute_view_iou_error(
        gt_cam, gt_dir,
        pred_cam, pred_dir,
        hfov=hfov_rad,
        vfov=vfov_rad,
        rc=rc,
        geom_id=int(mesh_id),
        tri_pts=tri_pts,
        tri_centroids=tri_centroids,
        tri_areas=tri_areas,
        near=0.05,
        far=None,
    )
    metrics.iou_error = iou_err
    if iou_val is not None and iou_err is not None:
        print(f"    view IoU: {iou_val:.3f} | IoU error: {iou_err:.3f}\n")
    else:
        print("    view IoU: n/a (missing direction or empty visibility)\n")

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
                      f"(H={math.degrees(hfov_rad):.0f}°, V={math.degrees(vfov_rad):.0f}°)")
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
        pred_sphere.paint_uniform_color([1.0, 0.9, 0.0])
        if not pred_sphere.has_vertex_normals():
            pred_sphere.compute_vertex_normals()
        vis.add_geometry("pred_cam", pred_sphere, material)
        vis.add_3d_label(pred_cam, f"Pred ({pred_source})")

        frustum_mat = None
        frustum_mat_pred = None
        frustum_gt = create_camera_frustum(gt_cam, gt_dir,
                                           colour=(1.0, 0.0, 0.0),
                                           h_fov=hfov_rad,
                                           v_fov=vfov_rad,
                                           scale=frustum_scale)
        frustum_pred = create_camera_frustum(pred_cam, pred_dir,
                                             colour=(1.0, 0.9, 0.0),
                                             h_fov=hfov_rad,
                                             v_fov=vfov_rad,
                                             scale=frustum_scale)
        if frustum_gt is not None:
            frustum_mat = rendering.MaterialRecord()
            frustum_mat.shader = "unlitLine"
            frustum_mat.line_width = 2.0
            frustum_mat.base_color = [1.0, 0.0, 0.0, 1.0]
            vis.add_geometry("frustum_gt", frustum_gt, frustum_mat)
        if frustum_pred is not None:
            frustum_mat_pred = rendering.MaterialRecord()
            frustum_mat_pred.shader = "unlitLine"
            frustum_mat_pred.line_width = 2.0
            frustum_mat_pred.base_color = [1.0, 0.9, 0.0, 1.0]
            vis.add_geometry("frustum_pred", frustum_pred, frustum_mat_pred)

        vis.reset_camera_to_default()
        gui.Application.instance.add_window(vis)

        # Second window: IoU-only overlays (GT-only, Pred-only, intersection)
        vis_iou = o3d.visualization.O3DVisualizer(f"{scene_id} – IoU overlap", 1280, 800)
        vis_iou.show_settings = False
        base_mat = rendering.MaterialRecord()
        base_mat.shader = "defaultLitTransparency"
        base_mat.base_color = [0.8, 0.8, 0.8, 0.18]
        vis_iou.add_geometry("mesh_base", mesh, base_mat)

        def _subset_mesh(base: o3d.geometry.TriangleMesh,
                         tris_idx: set[int],
                         colour: Tuple[float, float, float],
                         alpha: float) -> Optional[o3d.geometry.TriangleMesh]:
            """Create a coloured submesh from a set of triangle indices; None if empty."""
            if not tris_idx:
                return None

            idx_arr = np.asarray(sorted(tris_idx), dtype=np.int64)
            verts = np.asarray(base.vertices)
            tris_arr = np.asarray(base.triangles)[idx_arr]

            uniq, inv = np.unique(tris_arr.reshape(-1), return_inverse=True)
            new_verts = verts[uniq]
            new_tris = inv.reshape(-1, 3)

            sub = o3d.geometry.TriangleMesh()
            sub.vertices = o3d.utility.Vector3dVector(new_verts)
            sub.triangles = o3d.utility.Vector3iVector(new_tris)
            sub.paint_uniform_color(colour)
            if not sub.has_vertex_normals():
                sub.compute_vertex_normals()
            return sub

        gt_only = gt_vis_set - pred_vis_set
        pred_only = pred_vis_set - gt_vis_set
        both = gt_vis_set & pred_vis_set
        overlays = [
            ("iou_gt_only", gt_only, (1.0, 0.0, 0.0), 0.65),         # GT-only red
            ("iou_pred_only", pred_only, (1.0, 0.85, 0.0), 0.65),    # Pred-only yellow
            ("iou_both", both, (1.0, 0.4, 0.0), 0.85),               # Intersection orange
        ]
        for name, tri_set, colour, alpha in overlays:
            mesh_subset = _subset_mesh(mesh, tri_set, colour, alpha)
            if mesh_subset is None:
                continue
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLitTransparency"
            mat.base_color = [*colour, alpha]
            vis_iou.add_geometry(name, mesh_subset, mat)

        vis_iou.add_geometry("gt_cam_iou", gt_sphere, material)
        vis_iou.add_geometry("pred_cam_iou", pred_sphere, material)
        if frustum_gt is not None:
            vis_iou.add_geometry("frustum_gt_iou", frustum_gt, frustum_mat)
        if frustum_pred is not None:
            vis_iou.add_geometry("frustum_pred_iou", frustum_pred, frustum_mat_pred)
        vis_iou.reset_camera_to_default()
        gui.Application.instance.add_window(vis_iou)
        gui.Application.instance.run()

    return metrics


def main() -> None:
    args = parse_args()
    params_text = format_args_section(args)
    rng = np.random.default_rng(seed=args.seed)

    scenes = load_scene_graphs(args.graphs)

    candidate_ids = list(scenes.keys())
    if args.visualize_scene:
        if args.scene_ids:
            print("[WARN] --visualize_scene overrides --scene_ids.")
        if args.visualize_scene not in scenes:
            print(f"[ERROR] Requested scene '{args.visualize_scene}' not found in processed graphs.")
            return
        candidate_ids = [args.visualize_scene]
    elif args.scene_ids:
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
        if scene_metrics.iou_error is not None:
            print(f"    view IoU error: {scene_metrics.iou_error:.3f}")

    if not metrics_list:
        print("No scenes produced metrics. Nothing to report.")
        if args.log_file:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            payload = "No scenes produced metrics.\n\n" + params_text + "\n"
            args.log_file.write_text(payload)
            print(f"Empty summary logged to {args.log_file}")
        return

    table_text = build_metrics_table(metrics_list, args.hit_radius)
    if table_text:
        print("Scene-level summary table -------------------------------")
        print(table_text)
        print("---------------------------------------------------------\n")

    # Aggregate metrics
    def agg(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(np.median(arr))

    mean_gt, med_gt = agg([m.gt_prob for m in metrics_list])
    mean_nll, med_nll = agg([m.nll for m in metrics_list])
    mean_hit, med_hit = agg([m.hit_mass for m in metrics_list])
    mean_err, med_err = agg([m.distance_error for m in metrics_list])
    iou_err_values = [m.iou_error for m in metrics_list if m.iou_error is not None]
    mean_iou_err: Optional[float] = None
    med_iou_err: Optional[float] = None
    if iou_err_values:
        mean_iou_err, med_iou_err = agg([float(v) for v in iou_err_values])

    agg_lines = [
        "Aggregate metrics ---------------------------------------",
        f"  GT probability     : mean={mean_gt:.4f} | median={med_gt:.4f}",
        f"  NLL (surprisal)    : mean={mean_nll:.3f} | median={med_nll:.3f}",
        f"  Hit@{args.hit_radius:.2f}m       : mean={mean_hit:.3f} | median={med_hit:.3f}",
        f"  Distance error (m) : mean={mean_err:.3f} | median={med_err:.3f}",
        "---------------------------------------------------------\n",
    ]
    if mean_iou_err is not None and med_iou_err is not None:
        agg_lines.insert(-1,
                         f"  View IoU error     : mean={mean_iou_err:.3f} | median={med_iou_err:.3f}")
    print("\n".join(agg_lines))

    log_sections: List[str] = [params_text]
    if table_text:
        log_sections.append("Scene-level summary table")
        log_sections.append(table_text)
    log_sections.append("\n".join(agg_lines))
    log_payload = "\n\n".join(log_sections).rstrip() + "\n"
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text(log_payload)
        print(f"Metrics summary logged to {args.log_file}")

    if args.save_metrics:
        payload = [
            {
                "scene_id": m.scene_id,
                "frame_id": m.frame_id,
                "gt_prob": m.gt_prob,
                "nll": m.nll,
                "hit_mass": m.hit_mass,
                "distance_error": m.distance_error,
                "iou_error": m.iou_error,
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
                "iou_error": None if mean_iou_err is None else {"mean": mean_iou_err, "median": med_iou_err},
                "hit_radius": args.hit_radius,
                "top_k": args.top_k,
                "grid_step": args.grid_step,
            },
        }, indent=2))
        print(f"Metrics saved to {args.save_metrics}")


GUI_INITIALISED = False

if __name__ == "__main__":
    main()
