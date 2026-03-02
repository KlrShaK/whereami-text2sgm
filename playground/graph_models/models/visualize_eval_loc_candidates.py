#!/usr/bin/env python3
"""
visualize_eval_loc_candidates.py
--------------------------------
Generate candidate camera points/poses for ScanScribe-style evaluation frames.

For each scene, this script:
  1) Loads the per-frame JSON (visible_objects + scene_pose).
  2) Builds a caption scene graph and matches top-K 3D objects.
  3) Samples an XY grid at eye height and counts visible matched objects.
  4) Optionally builds FOV-weighted pose candidates (position + direction).
  5) Outputs grid candidates, GT pose, and predicted pose to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm isn't installed
    def tqdm(iterable, **kwargs):
        return iterable

# --------------------------------------------------------------------------- #
# Repository imports                                                         #
# --------------------------------------------------------------------------- #

sys.path.append("../data_processing")
sys.path.append("../../../")

from scene_graph import SceneGraph  # noqa: E402
from create_text_embeddings import create_embedding_nlp  # noqa: E402

# --------------------------------------------------------------------------- #
# Import helpers from visualize_loc_prob.py                                  #
# --------------------------------------------------------------------------- #

_THIS_DIR = Path(__file__).resolve().parent
_VIZ = _THIS_DIR / "visualize_loc_prob.py"
if not _VIZ.exists():
    raise FileNotFoundError(f"Expected visualize_loc_prob.py at {_VIZ}")

from importlib.machinery import SourceFileLoader

vizmod = SourceFileLoader("vizlocprob_candidates", str(_VIZ)).load_module()  # type: ignore

load_scene = vizmod.load_scene
topk_matched_objects = vizmod.topk_matched_objects
sample_grid = vizmod.sample_grid
first_hit_is_object = vizmod.first_hit_is_object
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
        return max(frames, key=lambda fs: len(fs.frame.get("visible_objects", {})))
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
# Camera pose + candidate helpers                                            #
# --------------------------------------------------------------------------- #

def camera_center_from_pose(pose: Iterable[Iterable[float]]) -> np.ndarray:
    mat = np.asarray(pose, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 scene_pose, got shape {mat.shape}")
    t = mat[:3, 3]
    return t.astype(np.float32)


def build_grid_candidates(cams: np.ndarray,
                          counts: np.ndarray,
                          probs: np.ndarray,
                          top_n: int) -> List[Dict[str, object]]:
    if top_n <= 0 or len(cams) == 0:
        return []
    order = np.argsort(-counts)
    top_idx = order[: min(top_n, len(order))]
    results: List[Dict[str, object]] = []
    for idx in top_idx:
        pos = cams[int(idx)]
        results.append({
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "visible_count": int(counts[int(idx)]),
            "prob": float(probs[int(idx)]),
        })
    return results


def build_pose_candidates(positions: List[np.ndarray],
                          directions: List[np.ndarray],
                          weights: List[float],
                          top_n: int) -> List[Dict[str, object]]:
    if top_n <= 0 or not positions or not weights:
        return []
    weights_np = np.asarray(weights, dtype=np.float64)
    order = np.argsort(-weights_np)
    top_idx = order[: min(top_n, len(order))]
    results: List[Dict[str, object]] = []
    for idx in top_idx:
        pos = positions[int(idx)]
        dir_vec = directions[int(idx)] if directions else None
        dir_out = None
        if dir_vec is not None:
            norm = float(np.linalg.norm(dir_vec))
            if norm > 1e-6:
                dir_out = (dir_vec / norm).tolist()
        results.append({
            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
            "direction": dir_out,
            "visible_count": int(weights_np[int(idx)]),
        })
    return results


def _cluster_weighted_prediction(positions: np.ndarray,
                                 weights: np.ndarray,
                                 bandwidth: float,
                                 max_points: int) -> Tuple[np.ndarray, List[int], np.ndarray]:
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

    return _cluster_weighted_prediction(positions,
                                        weights,
                                        bandwidth=bandwidth,
                                        max_points=max_points)


# --------------------------------------------------------------------------- #
# Main evaluation pipeline                                                   #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export candidate camera points/poses with GT and prediction."
    )
    parser.add_argument("--root", required=True,
                        help="Root directory containing <scene_id>/ meshes.")
    parser.add_argument("--graphs", required=True, type=Path,
                        help="processed_data directory holding 3dssg/*.pt files.")
    parser.add_argument("--query_root", type=Path,
                        help="Root containing per-scene output/descriptions/frame-*.json")

    parser.add_argument("--scene_ids", nargs="+",
                        help="Subset of scene IDs to evaluate. Defaults to intersection of graphs and query_root.")
    parser.add_argument("--max_scenes", type=int, default=300,
                        help="Limit number of scenes processed (after filtering). "
                             "Use 0 or a negative value to disable the limit.")
    parser.add_argument("--scene_sample_policy",
                        choices=["random", "first"],
                        default="random",
                        help="How to pick scenes when --max_scenes is smaller than available scenes.")
    parser.add_argument("--visualize_scene",
                        help="Scene ID to focus on. Overrides --scene_ids when set.")

    parser.add_argument("--frame_policy",
                        choices=["first", "index", "random", "max_visible", "max_pixels"],
                        default="max_visible",
                        help="Strategy to pick which frame JSON to evaluate per scene.")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Frame index used when --frame_policy=index.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for random scene/frame selection.")

    parser.add_argument("--top_k", type=int, default=25,
                        help="How many object matches to keep per caption.")
    parser.add_argument("--grid_step", type=float, default=0.25,
                        help="XY grid spacing in metres.")
    parser.add_argument("--eye_height", type=float, default=1.6,
                        help="Eye-height offset used by the grid sampler.")
    parser.add_argument("--prediction_strategy",
                        choices=["argmax", "random", "weighted"],
                        default="weighted",
                        help="How to convert candidate positions into a final camera prediction.")
    parser.add_argument("--cluster_bandwidth", type=float, default=0.75,
                        help="Bandwidth (metres) for the weighted cluster strategy.")
    parser.add_argument("--max_cluster_points", type=int, default=512,
                        help="Maximum candidates used when computing cluster-aware predictions.")

    parser.add_argument("--h_fov_deg", type=float, default=100.0,
                        help="Horizontal FOV (degrees) for pose candidates.")
    parser.add_argument("--v_fov_deg", type=float, default=60.0,
                        help="Vertical FOV (degrees) for pose candidates.")
    parser.add_argument("--arrow_stride", type=int, default=2,
                        help="Evaluate every Nth grid camera when building pose candidates.")

    parser.add_argument("--top_candidates", type=int, default=10,
                        help="Number of top grid candidates to export.")
    parser.add_argument("--top_pose_candidates", type=int, default=10,
                        help="Number of top FOV pose candidates to export.")
    parser.add_argument("--output_json", type=Path,
                        help="Path to write JSON output. Defaults to eval/eval_pose_candidates.json")
    return parser.parse_args()


def load_scene_graphs(graphs_dir: Path) -> Dict[str, SceneGraph]:
    g3d_path = graphs_dir / "3dssg" / "3dssg_graphs_processed_edgelists_relationembed.pt"
    if not g3d_path.exists():
        raise FileNotFoundError(g3d_path)

    # PyTorch >=2.6 defaults to weights_only=True, which breaks loading this graph dict.
    g3d = torch.load(g3d_path, map_location="cpu", weights_only=False)
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
                   rng: np.random.Generator) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    mesh_root = Path(args.root)
    scene_dir = mesh_root / scene_id
    if not scene_dir.exists():
        return None, "scene directory missing"

    query_root = ensure_query_root(args.query_root, Path(args.root))
    desc_dir = query_root / scene_id / "output" / "descriptions"
    if not desc_dir.exists():
        desc_dir = scene_dir / "output" / "descriptions"
    frames = load_frame_jsons(desc_dir)
    if not frames:
        return None, f"no frame JSONs under {desc_dir}"

    selection = select_frame(frames, args.frame_policy, args.frame_index, rng)
    if selection is None:
        return None, "frame selection failed"

    frame = selection.frame
    try:
        caption_graph, _ = frame_to_scenegraph(frame)
    except Exception as exc:  # noqa: BLE001
        return None, f"caption graph failed: {exc}"

    gt_pose = frame.get("scene_pose")
    if gt_pose is None:
        return None, "scene_pose missing"

    pose_mat = np.asarray(gt_pose, dtype=np.float64)
    gt_cam = camera_center_from_pose(pose_mat)
    rot_cam_world = pose_mat[:3, :3]
    forward_cv = rot_cam_world @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm_forward = np.linalg.norm(forward_cv)
    gt_dir = forward_cv / norm_forward if norm_forward > 1e-6 else None

    obj_ids = topk_matched_objects(caption_graph, scene_graph, k=args.top_k)
    if not obj_ids:
        return None, "no cosine matches"

    mesh, tri2obj, obj2faces = load_scene(scene_dir)
    rc = o3d.t.geometry.RaycastingScene()
    rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    verts = np.asarray(mesh.vertices)
    cams = sample_grid(verts, step=args.grid_step, z_eye=args.eye_height)

    xs, ys = verts[:, 0], verts[:, 1]
    gx = np.arange(xs.min(), xs.max() + 1e-4, args.grid_step)
    gy = np.arange(ys.min(), ys.max() + 1e-4, args.grid_step)
    nx, ny = len(gx), len(gy)

    tris = np.asarray(mesh.triangles)
    centroids: Dict[int, np.ndarray] = {}
    for oid in obj_ids:
        faces = obj2faces.get(int(oid))
        if faces is None or not len(faces):
            continue
        centroids[int(oid)] = verts[np.unique(tris[faces].ravel())].mean(axis=0)

    if not centroids:
        return None, "matched objects missing geometry"

    visible_dirs: List[List[np.ndarray]] = [[] for _ in range(len(cams))]
    counts = np.zeros(len(cams), dtype=np.int32)
    for idx, cam in enumerate(cams):
        count = 0
        for oid, centre in centroids.items():
            if first_hit_is_object(cam, centre, oid, rc, tri2obj):
                count += 1
                d = centre - cam
                l = np.linalg.norm(d)
                if l > 1e-6:
                    visible_dirs[idx].append(d / l)
        counts[idx] = count

    total = int(counts.sum())
    if total == 0:
        return None, "matched objects invisible from grid"
    probs = counts / float(total)

    grid_point_candidates = build_grid_candidates(cams, counts, probs, args.top_candidates)
    max_visible_count = int(counts.max())
    max_visible_points = int(np.sum(counts == max_visible_count))

    arrow_positions: List[np.ndarray] = []
    arrow_dirs: List[np.ndarray] = []
    arrow_weights: List[float] = []
    have_arrow_helpers = bool(dir_to_yaw_pitch and best_fov_window and average_direction)
    if have_arrow_helpers:
        hfov = math.radians(args.h_fov_deg)
        vfov = math.radians(args.v_fov_deg)
        stride = max(1, int(args.arrow_stride))
        for gy_i in range(0, ny, stride):
            for gx_i in range(0, nx, stride):
                idx = gy_i * nx + gx_i
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

    fov_pose_candidates = build_pose_candidates(arrow_positions,
                                                arrow_dirs,
                                                arrow_weights,
                                                args.top_pose_candidates)

    candidate_dirs: Optional[np.ndarray] = None
    candidate_source = "grid_visibility"
    if arrow_weights:
        candidate_source = "arrow_fov"
        candidate_positions = np.asarray(arrow_positions, dtype=np.float64)
        candidate_weights = np.asarray(arrow_weights, dtype=np.float64)
        candidate_dirs = np.asarray(arrow_dirs, dtype=np.float64)
    else:
        candidate_positions = cams
        candidate_weights = counts.astype(np.float64)

    pred_dir: Optional[np.ndarray] = None
    pred_cam, selection_idx, selection_weights = select_prediction_point(
        candidate_positions,
        candidate_weights,
        strategy=args.prediction_strategy,
        rng=rng,
        bandwidth=args.cluster_bandwidth,
        max_points=args.max_cluster_points,
    )

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

    frame_id = str(frame.get("image_index", selection.path.name))
    record: Dict[str, object] = {
        "scene_id": scene_id,
        "frame_id": frame_id,
        "frame_path": str(selection.path),
        "description": frame.get("description"),
        "timestamp": frame.get("timestamp"),
        "visible_objects_count": len(frame.get("visible_objects", {}) or {}),
        "matched_object_ids": [int(o) for o in obj_ids],
        "matched_object_count": len(obj_ids),
        "grid": {
            "step": args.grid_step,
            "eye_height": args.eye_height,
            "points": len(cams),
        },
        "grid_max_visible_count": max_visible_count,
        "grid_max_visible_point_count": max_visible_points,
        "grid_point_candidates": grid_point_candidates,
        "fov_pose_candidates": fov_pose_candidates,
        "gt_pose": {
            "position": gt_cam.tolist(),
            "direction": gt_dir.tolist() if gt_dir is not None else None,
            "scene_pose": pose_mat.tolist(),
        },
        "predicted_pose": {
            "position": pred_cam.tolist(),
            "direction": pred_dir.tolist() if pred_dir is not None else None,
            "source": f"{candidate_source}:{args.prediction_strategy}",
        },
    }
    return record, None


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(seed=args.seed)

    print(f"Loading graphs from {args.graphs} ...")
    scenes = load_scene_graphs(args.graphs)

    candidate_ids = list(scenes.keys())
    if args.visualize_scene:
        if args.visualize_scene not in scenes:
            print(f"[ERROR] Requested scene '{args.visualize_scene}' not found in processed graphs.")
            return
        candidate_ids = [args.visualize_scene]
    elif args.scene_ids:
        scene_set = set(args.scene_ids)
        candidate_ids = [sid for sid in candidate_ids if sid in scene_set]
    else:
        query_root = ensure_query_root(args.query_root, Path(args.root))
        candidate_ids = [
            sid for sid in candidate_ids
            if (query_root / sid / "output" / "descriptions").exists()
            or (Path(args.root) / sid / "output" / "descriptions").exists()
        ]

    candidate_ids.sort()
    if args.max_scenes is not None and args.max_scenes > 0 and len(candidate_ids) > args.max_scenes:
        if args.scene_sample_policy == "random":
            sample_idx = rng.choice(len(candidate_ids), size=args.max_scenes, replace=False)
            candidate_ids = [candidate_ids[int(i)] for i in sample_idx]
            candidate_ids.sort()
        else:
            candidate_ids = candidate_ids[: args.max_scenes]

    if args.output_json is None:
        args.output_json = _THIS_DIR / "eval" / "eval_pose_candidates.json"

    results: List[Dict[str, object]] = []
    skipped: List[Dict[str, str]] = []
    for sid in tqdm(candidate_ids, desc="Scenes", unit="scene"):
        record, reason = evaluate_scene(sid, scenes[sid], args, rng)
        if record is None:
            skipped.append({"scene_id": sid, "reason": reason or "unknown"})
            continue
        results.append(record)

    payload = {
        "scenes": results,
        "skipped": skipped,
        "args": {
            "root": str(args.root),
            "graphs": str(args.graphs),
            "query_root": str(args.query_root) if args.query_root else None,
            "frame_policy": args.frame_policy,
            "frame_index": args.frame_index,
            "seed": args.seed,
            "max_scenes": args.max_scenes,
            "scene_sample_policy": args.scene_sample_policy,
            "top_k": args.top_k,
            "grid_step": args.grid_step,
            "eye_height": args.eye_height,
            "prediction_strategy": args.prediction_strategy,
            "cluster_bandwidth": args.cluster_bandwidth,
            "max_cluster_points": args.max_cluster_points,
            "h_fov_deg": args.h_fov_deg,
            "v_fov_deg": args.v_fov_deg,
            "arrow_stride": args.arrow_stride,
            "top_candidates": args.top_candidates,
            "top_pose_candidates": args.top_pose_candidates,
        },
        "selected_scene_ids": candidate_ids,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote candidate poses to {output_path}")


if __name__ == "__main__":
    main()
