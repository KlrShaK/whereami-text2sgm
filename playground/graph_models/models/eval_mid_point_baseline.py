#!/usr/bin/env python3
"""
eval_mid_point_baseline.py
--------------------------
Simple localisation baseline:
1) place camera at the midpoint of the scene floor/bounds,
2) sample a random viewing direction,
3) compute the same evaluation metrics as the main localisation evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from visualize_eval_loc_mk4 import (  # noqa: E402
    SceneMetrics,
    _extract_floor_bbox,
    build_metrics_table,
    camera_center_from_pose,
    compute_metrics,
    compute_view_iou_error,
    ensure_query_root,
    format_args_section,
    load_frame_jsons,
    load_scene,
    load_scene_graphs,
    select_frame,
)


def random_forward(rng: np.random.Generator, max_pitch_deg: float) -> np.ndarray:
    """Sample a random unit direction from yaw + bounded pitch."""
    yaw = float(rng.uniform(-math.pi, math.pi))
    max_pitch = math.radians(float(max_pitch_deg))
    pitch = float(rng.uniform(-max_pitch, max_pitch))
    cp = math.cos(pitch)
    direction = np.array([
        cp * math.cos(yaw),
        cp * math.sin(yaw),
        math.sin(pitch),
    ], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return direction / norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mid-point + random-pose localisation baseline evaluator."
    )
    parser.add_argument("--root", required=True,
                        help="Root directory containing <scene_id>/ meshes.")
    parser.add_argument("--graphs", required=True, type=Path,
                        help="processed_data directory holding 3dssg/*.pt files.")
    parser.add_argument("--query_root", type=Path,
                        help="Root containing per-scene output/descriptions/frame-*.json")
    parser.add_argument("--scene_ids", nargs="+",
                        help="Subset of scene IDs to evaluate.")
    parser.add_argument("--max_scenes", type=int,
                        help="Limit number of scenes processed.")
    parser.add_argument("--visualize_scene",
                        help="Scene ID to evaluate alone. Overrides --scene_ids when set.")
    parser.add_argument("--frame_policy",
                        choices=["first", "index", "random", "max_visible", "max_pixels"],
                        default="max_visible",
                        help="Strategy for selecting a frame JSON per scene.")
    parser.add_argument("--frame_index", type=int, default=0,
                        help="Frame index used when --frame_policy=index.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for frame selection and random pose.")
    parser.add_argument("--eye_height", type=float, default=1.6,
                        help="Eye-height offset from floor/bounds.")
    parser.add_argument("--random_pitch_deg", type=float, default=30.0,
                        help="Random pitch sampled uniformly in [-deg, +deg].")

    parser.add_argument("--hit_radii", nargs="+", type=float,
                        default=[0.75, 1.0, 1.5, 2.0, 2.5],
                        help="Radii (metres) for Hit@r mass curve.")
    parser.add_argument("--mass_percentiles", nargs="+", type=float,
                        default=[50.0, 90.0],
                        help="Percentiles for mass-radius metric.")
    parser.add_argument("--top_k_min_dist", type=int, default=10,
                        help="K for Top-K min distance metric.")
    parser.add_argument("--h_fov_deg", type=float, default=39.31,
                        help="Horizontal FOV (degrees) for IoU evaluation.")
    parser.add_argument("--v_fov_deg", type=float, default=64.76,
                        help="Vertical FOV (degrees) for IoU evaluation.")

    parser.add_argument("--save_metrics", type=Path,
                        default=Path("eval/baseline_eval_metrics_mid_point.json"),
                        help="Path to save per-scene metrics JSON.")
    parser.add_argument("--log_file", type=Path,
                        default=Path("eval/baseline_eval_loc_summary_mid_point.log"),
                        help="Path to write summary log.")
    return parser.parse_args()


def evaluate_scene(scene_id: str,
                   args: argparse.Namespace,
                   rng: np.random.Generator) -> Optional[SceneMetrics]:
    scene_dir = Path(args.root) / scene_id
    if not scene_dir.exists():
        print(f"[WARN] Scene directory missing for {scene_id} — skipped.")
        return None

    query_root = ensure_query_root(args.query_root, Path(args.root))
    desc_dir = query_root / scene_id / "output" / "descriptions"
    if not desc_dir.exists():
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

    gt_pose = frame.get("scene_pose")
    if gt_pose is None:
        print(f"[WARN] scene_pose missing in {selection.path} — skipped.")
        return None

    pose_mat = np.asarray(gt_pose, dtype=np.float64)
    gt_cam = camera_center_from_pose(pose_mat)
    rot_cam_world = pose_mat[:3, :3]
    gt_forward = rot_cam_world @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    gt_forward_norm = float(np.linalg.norm(gt_forward))
    gt_dir = gt_forward / gt_forward_norm if gt_forward_norm > 1e-6 else None

    mesh, _tri2obj, obj2faces = load_scene(scene_dir)
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

    floor_bbox = _extract_floor_bbox(scene_dir, verts, tris, obj2faces)
    if floor_bbox is not None:
        x_mid = 0.5 * (floor_bbox["x_min"] + floor_bbox["x_max"])
        y_mid = 0.5 * (floor_bbox["y_min"] + floor_bbox["y_max"])
        z_eye = floor_bbox["z_max"] + args.eye_height
        bounds_msg = (f"floor bbox center=({x_mid:.2f}, {y_mid:.2f}) "
                      f"z_eye={z_eye:.2f} m")
    else:
        x_mid = 0.5 * float(verts[:, 0].min() + verts[:, 0].max())
        y_mid = 0.5 * float(verts[:, 1].min() + verts[:, 1].max())
        z_eye = float(verts[:, 2].min()) + args.eye_height
        bounds_msg = (f"mesh bbox center=({x_mid:.2f}, {y_mid:.2f}) "
                      f"z_eye={z_eye:.2f} m")

    pred_cam = np.array([x_mid, y_mid, z_eye], dtype=np.float64)
    pred_dir = random_forward(rng, args.random_pitch_deg)

    cams = pred_cam.reshape(1, 3)
    probs = np.array([1.0], dtype=np.float64)
    _pred_idx, metrics = compute_metrics(cams, probs, gt_cam,
                                         hit_radii=args.hit_radii,
                                         mass_percentiles=args.mass_percentiles,
                                         topk_k=args.top_k_min_dist)
    metrics.scene_id = scene_id
    metrics.frame_id = str(frame.get("image_index", selection.path.name))
    metrics.matched_objects = 0
    metrics.distance_error = float(np.linalg.norm(pred_cam - gt_cam))

    if gt_dir is not None:
        dot = float(np.clip(np.dot(gt_dir, pred_dir), -1.0, 1.0))
        metrics.angular_error_deg = float(math.degrees(math.acos(dot)))
    else:
        metrics.angular_error_deg = None

    hfov = math.radians(args.h_fov_deg)
    vfov = math.radians(args.v_fov_deg)
    iou_val, iou_err, _gt_set, _pred_set = compute_view_iou_error(
        gt_cam, gt_dir, pred_cam, pred_dir,
        hfov=hfov, vfov=vfov,
        rc=rc, geom_id=int(mesh_id),
        tri_pts=tri_pts, tri_centroids=tri_centroids, tri_areas=tri_areas,
        near=0.05, far=None,
    )
    metrics.iou_error = iou_err

    print(f"    baseline pose: {bounds_msg}")
    print(f"    predicted camera (midpoint): {pred_cam.tolist()} | "
          f"err={metrics.distance_error:.3f} m")
    print(f"    predicted direction (random): {pred_dir.tolist()}")
    if iou_val is not None and iou_err is not None:
        print(f"    view IoU: {iou_val:.3f} | IoU error: {iou_err:.3f}\n")
    else:
        print("    view IoU: n/a (missing direction or empty visibility)\n")

    return metrics


def main() -> None:
    args = parse_args()
    args.hit_radii = [float(r) for r in args.hit_radii]
    args.mass_percentiles = [float(p) for p in args.mass_percentiles]
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
        query_root = ensure_query_root(args.query_root, Path(args.root))
        candidate_ids = [
            sid for sid in candidate_ids
            if (query_root / sid / "output" / "descriptions").exists()
            or (Path(args.root) / sid / "output" / "descriptions").exists()
        ]

    candidate_ids.sort()
    if args.max_scenes is not None:
        candidate_ids = candidate_ids[: args.max_scenes]

    print(f"Evaluating midpoint baseline on {len(candidate_ids)} scene(s)...\n")

    metrics_list: List[SceneMetrics] = []
    for idx, sid in enumerate(candidate_ids, start=1):
        print(f"[{idx:03d}/{len(candidate_ids):03d}] {sid}")
        scene_metrics = evaluate_scene(sid, args, rng)
        if scene_metrics is None:
            continue
        metrics_list.append(scene_metrics)
        print(f"    frame: {scene_metrics.frame_id}")
        print(f"    matches: {scene_metrics.matched_objects} | grid pts: {scene_metrics.grid_points}")
        hit_line = " | ".join(
            f"hit@{r:.2f}m: {scene_metrics.hit_masses.get(r, 0.0):.3f}"
            for r in sorted(scene_metrics.hit_masses)
        )
        print(f"    {hit_line}")
        mass_line = " | ".join(
            f"R{p:.0f}%: {scene_metrics.mass_radii.get(p, float('nan')):.3f} m"
            for p in sorted(scene_metrics.mass_radii)
        )
        if mass_line:
            print(f"    mass-radius: {mass_line}")
        ang_err = ("n/a" if scene_metrics.angular_error_deg is None
                   else f"{scene_metrics.angular_error_deg:.2f}°")
        print(f"    topK{args.top_k_min_dist} min dist: {scene_metrics.topk_min_dist:.3f} m | "
              f"dist_err: {scene_metrics.distance_error:.3f} m | ang_err: {ang_err}\n")
        if scene_metrics.iou_error is not None:
            print(f"    view IoU error: {scene_metrics.iou_error:.3f}")

    if not metrics_list:
        print("No scenes produced metrics. Nothing to report.")
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        payload = "No scenes produced metrics.\n\n" + params_text + "\n"
        args.log_file.write_text(payload)
        print(f"Empty summary logged to {args.log_file}")
        return

    table_text = build_metrics_table(metrics_list,
                                     args.hit_radii,
                                     args.mass_percentiles,
                                     args.top_k_min_dist)
    if table_text:
        print("Scene-level summary table -------------------------------")
        print(table_text)
        print("---------------------------------------------------------\n")

    def agg(values: List[float]) -> Tuple[float, float]:
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(np.median(arr))

    hit_stats: Dict[float, Tuple[float, float]] = {}
    for r in sorted(set(args.hit_radii)):
        vals = [m.hit_masses.get(r, 0.0) for m in metrics_list]
        hit_stats[r] = agg(vals)

    mass_radius_stats: Dict[float, Tuple[float, float]] = {}
    for p in sorted(set(args.mass_percentiles)):
        vals = [m.mass_radii.get(p, float("nan")) for m in metrics_list]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            mass_radius_stats[p] = agg(vals)

    mean_topk, med_topk = agg([m.topk_min_dist for m in metrics_list])
    mean_err, med_err = agg([m.distance_error for m in metrics_list])
    ang_values = [m.angular_error_deg for m in metrics_list if m.angular_error_deg is not None]
    mean_ang: Optional[float] = None
    med_ang: Optional[float] = None
    if ang_values:
        mean_ang, med_ang = agg([float(v) for v in ang_values])
    iou_err_values = [m.iou_error for m in metrics_list if m.iou_error is not None]
    mean_iou_err: Optional[float] = None
    med_iou_err: Optional[float] = None
    if iou_err_values:
        mean_iou_err, med_iou_err = agg([float(v) for v in iou_err_values])

    agg_lines = [
        "Aggregate metrics ---------------------------------------",
        f"  TopK{args.top_k_min_dist} min dist (m): mean={mean_topk:.3f} | median={med_topk:.3f}",
        f"  Distance error (m)      : mean={mean_err:.3f} | median={med_err:.3f}",
        "---------------------------------------------------------\n",
    ]
    for r in sorted(hit_stats):
        mean_hit, med_hit = hit_stats[r]
        agg_lines.insert(-1, f"  Hit@{r:.2f}m              : mean={mean_hit:.3f} | median={med_hit:.3f}")
    for p in sorted(mass_radius_stats):
        mean_r, med_r = mass_radius_stats[p]
        agg_lines.insert(-1, f"  Mass-radius R{p:.0f}% (m): mean={mean_r:.3f} | median={med_r:.3f}")
    if mean_ang is not None and med_ang is not None:
        agg_lines.insert(-1, f"  Angular error (deg)   : mean={mean_ang:.2f} | median={med_ang:.2f}")
    if mean_iou_err is not None and med_iou_err is not None:
        agg_lines.insert(-1, f"  View IoU error        : mean={mean_iou_err:.3f} | median={med_iou_err:.3f}")
    print("\n".join(agg_lines))

    log_sections: List[str] = [params_text]
    if table_text:
        log_sections.extend(["Scene-level summary table", table_text])
    log_sections.append("\n".join(agg_lines))
    log_payload = "\n\n".join(log_sections).rstrip() + "\n"
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text(log_payload)
    print(f"Metrics summary logged to {args.log_file}")

    args.save_metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "scene_id": m.scene_id,
            "frame_id": m.frame_id,
            "hit_masses": {str(k): v for k, v in m.hit_masses.items()},
            "mass_radii": {str(k): v for k, v in m.mass_radii.items()},
            "topk_min_dist": m.topk_min_dist,
            "distance_error": m.distance_error,
            "angular_error_deg": m.angular_error_deg,
            "grid_points": m.grid_points,
            "matched_objects": m.matched_objects,
            "iou_error": m.iou_error,
        }
        for m in metrics_list
    ]
    args.save_metrics.write_text(json.dumps(payload, indent=2))
    print(f"Per-scene metrics saved to {args.save_metrics}")


if __name__ == "__main__":
    main()
