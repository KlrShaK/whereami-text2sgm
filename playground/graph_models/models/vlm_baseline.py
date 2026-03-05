#!/usr/bin/env python3
"""
VLM baseline evaluator for topdown pose prediction on 3RScan_processed.

Pipeline per frame:
1) Read topdown image and textual frame description.
2) Query Qwen-VL for topdown pixel (u,v) and heading angle (w).
3) Project prediction to world coordinates using topdown camera intrinsics/extrinsics.
4) Compute Euclidean and angular errors against frame ground-truth pose.
5) Save per-frame metrics JSON and a summary log under eval/.

Visualization is debug-only and fully gated behind --visualize.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - optional dependency in some environments.
    o3d = None


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_TOPDOWN_NAME = "topdown.png"

PREFERRED_MESH_FILES = (
    "mesh.refined.v2.obj",
    "labels.instances.annotated.v2.ply",
    "mesh.refined.ply",
    "mesh.refined.obj",
    "mesh.obj",
)


@dataclass(frozen=True)
class Pose3D:
    position: np.ndarray
    direction: Optional[np.ndarray]


@dataclass(frozen=True)
class QwenModelBundle:
    model: Qwen3VLForConditionalGeneration
    processor: AutoProcessor
    patch_size: int
    merge_size: int


@dataclass(frozen=True)
class FailureRecord:
    scene_id: str
    frame_id: str
    reason: str


@dataclass(frozen=True)
class SceneIoUContext:
    raycasting_scene: "o3d.t.geometry.RaycastingScene"
    mesh_id: int
    tri_points: np.ndarray
    tri_centroids: np.ndarray
    tri_areas: np.ndarray


def normalize(v: np.ndarray, eps: float = 1e-9) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return None
    return v / norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen VLM baseline on all scenes/frames and evaluate pose errors."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/media/klrshak/Backup/Datasets/3RScan_processed"),
        help="Root directory containing <scene_id>/ meshes (and optionally topdown/frames).",
    )
    parser.add_argument(
        "--dataset",
        choices=["3rscan", "scannet"],
        default="3rscan",
        help="Dataset layout: 3rscan or scannet.",
    )
    parser.add_argument(
        "--query_root",
        type=Path,
        default=None,
        help="Root containing per-scene topdown images and output/descriptions/*.json. "
             "Defaults to --root when omitted.",
    )
    parser.add_argument(
        "--save_metrics",
        type=Path,
        default=Path("eval/baseline_eval_metric_qwen.json"),
        help="Path to write per-frame metrics JSON.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=Path("eval/baseline_eval_metric_qwen.log"),
        help="Path to write aggregate summary log.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="HF model id for Qwen VLM.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda",
        help="device_map passed to model.from_pretrained.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens for VLM generation.",
    )
    parser.add_argument(
        "--retry_count",
        type=int,
        default=2,
        help="Retry attempts after a failed frame inference/parse.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing metrics JSON if present.",
    )
    parser.add_argument(
        "--scene_ids",
        nargs="+",
        help="Optional list of scene IDs to evaluate.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        help="Optional maximum number of scenes to process.",
    )
    parser.add_argument(
        "--max_frames_per_scene",
        type=int,
        help="Optional maximum frames per scene (for debug runs).",
    )
    parser.add_argument(
        "--frame_policy",
        choices=["first", "index", "random", "max_visible", "max_pixels", "all"],
        default="all",
        help="Strategy for selecting frame JSON(s) per scene. "
             "'all' evaluates every frame in the scene (default).",
    )
    parser.add_argument(
        "--frame_index",
        type=int,
        default=0,
        help="Frame index used when --frame_policy=index.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic behavior where relevant.",
    )
    parser.add_argument(
        "--eye_height_m",
        type=float,
        default=1.6,
        help="Eye height above floor plane for predicted camera position.",
    )
    parser.add_argument(
        "--direction_step_px",
        type=float,
        default=20.0,
        help="Pixel step used to convert heading angle to world direction.",
    )
    parser.add_argument(
        "--hit_radii",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.5, 2.0, 2.5],
        help="Radii (m) for Hit@r metrics.",
    )
    parser.add_argument(
        "--recall_radii",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.5, 2.0],
        help="Radii (m) for binary Recall@r on final prediction.",
    )
    parser.add_argument(
        "--mass_percentiles",
        nargs="+",
        type=float,
        default=[50.0, 90.0],
        help="Mass-radius percentiles (single-point baseline maps to distance).",
    )
    parser.add_argument(
        "--top_k_min_dist",
        type=int,
        default=10,
        help="Top-K min distance setting (single-point baseline maps to distance).",
    )
    parser.add_argument(
        "--h_fov_deg",
        type=float,
        default=39.31,
        help="Horizontal FOV (degrees) for IoU evaluation.",
    )
    parser.add_argument(
        "--v_fov_deg",
        type=float,
        default=64.76,
        help="Vertical FOV (degrees) for IoU evaluation.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Debug only: visualize mesh with GT/predicted pose for each processed frame.",
    )
    return parser.parse_args()


def format_args_section(args: argparse.Namespace) -> str:
    lines = ["Parameters used", "---------------"]
    for key in sorted(vars(args)):
        value = getattr(args, key)
        if isinstance(value, Path):
            value = str(value)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_prompt(scene_description: str, width: int, height: int) -> str:
    return f"""
You are a vision model helping with top-down localization and heading estimation.

Return ONLY a JSON object with keys u,v,w:
{{"u": <float>, "v": <float>, "w": <float>}}

Coordinate convention:
- Origin (0,0) is top-left of the image.
- u increases to the right, v increases downward.
- u must be in [0, {width}), v must be in [0, {height}).

Angle convention:
- w in degrees in [0,360)
- w = 0° points right (+u)
- w increases CCW (90° up, 180° left, 270° down)

IMPORTANT:
- u,v must refer to pixel coordinates of the ORIGINAL input image size {width}x{height}.

Scene description:
<<<
{scene_description}
>>>
""".strip()


def parse_prediction_json(raw_response: str) -> Dict[str, float]:
    start = raw_response.find("{")
    end = raw_response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    parsed = json.loads(raw_response[start:end + 1])
    for key in ("u", "v", "w"):
        if key not in parsed or not isinstance(parsed[key], (int, float)):
            raise ValueError(f"Model output missing numeric '{key}'.")
    return {"u": float(parsed["u"]), "v": float(parsed["v"]), "w": float(parsed["w"])}


def clamp_prediction(u: float, v: float, w: float, width: int, height: int) -> Tuple[int, int, float]:
    u_i = int(round(u))
    v_i = int(round(v))
    if not (0 <= u_i < width):
        u_i = max(0, min(u_i, width - 1))
    if not (0 <= v_i < height):
        v_i = max(0, min(v_i, height - 1))
    w_norm = float(w % 360.0)
    return u_i, v_i, w_norm


def ensure_processor_multiple(images: Iterable[Image.Image], patch_size: int, merge_size: int) -> None:
    required = patch_size * merge_size
    for idx, image in enumerate(images):
        w_seen, h_seen = image.size
        if (w_seen % required) != 0 or (h_seen % required) != 0:
            raise ValueError(
                "Vision preprocessing mismatch: image "
                f"index {idx} resized to {w_seen}x{h_seen}, not divisible by {required}."
            )


def load_qwen_bundle(args: argparse.Namespace) -> QwenModelBundle:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map=args.device_map,
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, do_resize=False, use_fast=False)
    patch_size = int(processor.image_processor.patch_size)
    merge_size = int(processor.image_processor.merge_size)
    return QwenModelBundle(model=model, processor=processor, patch_size=patch_size, merge_size=merge_size)


def run_qwen_inference(
    bundle: QwenModelBundle,
    image_path: Path,
    description: str,
    max_new_tokens: int,
) -> Tuple[int, int, float, str]:
    with Image.open(image_path) as img:
        width, height = img.convert("RGB").size

    prompt = build_prompt(description, width, height)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=bundle.patch_size)
    image_seq: List[Image.Image]
    if isinstance(image_inputs, (list, tuple)):
        image_seq = list(image_inputs)
    else:
        image_seq = [image_inputs]
    ensure_processor_multiple(image_seq, bundle.patch_size, bundle.merge_size)

    seen = image_seq[0]
    seen_w, seen_h = seen.size

    text = bundle.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = bundle.processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(bundle.model.device)
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)

    with torch.no_grad():
        out = bundle.model.generate(**inputs, generation_config=gen_cfg)
    generated = out[0][inputs["input_ids"].shape[-1]:]
    raw_response = bundle.processor.decode(generated, skip_special_tokens=True).strip()
    pred = parse_prediction_json(raw_response)
    # Heuristic: exact all-zero output is usually a collapsed/invalid prediction.
    # Raise so the caller's retry loop can request another generation attempt.
    if pred["u"] == 0.0 and pred["v"] == 0.0 and pred["w"] == 0.0:
        raise ValueError("Degenerate model output (u=v=w=0).")

    u = pred["u"]
    v = pred["v"]
    if seen_w != width or seen_h != height:
        u *= width / float(seen_w)
        v *= height / float(seen_h)
    u_i, v_i, w = clamp_prediction(u, v, pred["w"], width, height)
    return u_i, v_i, w, raw_response


def find_scene_dirs(root: Path, scene_ids: Optional[Sequence[str]], max_scenes: Optional[int]) -> List[Path]:
    candidates = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if scene_ids:
        wanted = set(scene_ids)
        candidates = [p for p in candidates if p.name in wanted]
    if max_scenes is not None:
        candidates = candidates[:max_scenes]
    return candidates


def resolve_topdown_paths(
    scene_dir: Path,
    query_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Locate topdown image + camera npz.  Checks query_root/<scene>/ first,
    then falls back to scene_dir/."""
    candidates = [scene_dir]
    if query_root is not None:
        qr_scene = query_root / scene_dir.name
        if qr_scene.exists():
            candidates.insert(0, qr_scene)
    for base in candidates:
        image_path = base / DEFAULT_TOPDOWN_NAME
        camera_path = base / "topdown_camera.npz"
        if image_path.exists() and camera_path.exists():
            return image_path, camera_path
    raise FileNotFoundError(
        f"Missing topdown image or camera sidecar for scene '{scene_dir.name}'. "
        f"Searched: {[str(c) for c in candidates]}"
    )


def frame_json_paths(
    scene_dir: Path,
    max_frames: Optional[int],
    query_root: Optional[Path] = None,
) -> List[Path]:
    """Discover frame JSONs.  Checks query_root/<scene>/output/descriptions
    first, then falls back to scene_dir/output/descriptions."""
    qr = query_root if query_root is not None else scene_dir
    desc_dir = qr / scene_dir.name / "output" / "descriptions"
    if not desc_dir.exists():
        desc_dir = scene_dir / "output" / "descriptions"
    frames = [
        p
        for p in sorted(desc_dir.glob("*.json"))
        if not p.stem.endswith("_parsed") and p.name != "all_descriptions.json"
    ]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def load_topdown_camera(camera_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(camera_path)
    intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
    extrinsic = np.asarray(data["extrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic 3x3 in {camera_path}, got {intrinsic.shape}")
    if extrinsic.shape != (4, 4):
        raise ValueError(f"Expected extrinsic 4x4 in {camera_path}, got {extrinsic.shape}")
    return intrinsic, extrinsic


def gt_pose_from_frame(frame: dict, frame_path: Path) -> Pose3D:
    pose = np.asarray(frame.get("scene_pose"), dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected scene_pose 4x4 in {frame_path}, got {pose.shape}")
    gt_pos = pose[:3, 3].astype(np.float64)
    gt_dir = pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    gt_dir = normalize(gt_dir)
    return Pose3D(position=gt_pos, direction=gt_dir)


def estimate_floor_z(frame: dict, gt_pose: Pose3D, eye_height_m: float) -> float:
    visible = frame.get("visible_objects", {}) or {}
    floor_lows: List[float] = []
    for obj in visible.values():
        label = str(obj.get("label", "")).strip().lower()
        if "floor" not in label:
            continue
        bbox = obj.get("bbox_world")
        if not isinstance(bbox, list) or len(bbox) != 2:
            continue
        try:
            lo = float(bbox[0][2])
            hi = float(bbox[1][2])
        except (TypeError, ValueError, IndexError):
            continue
        floor_lows.append(min(lo, hi))

    if floor_lows:
        return float(np.median(np.asarray(floor_lows, dtype=np.float64)))

    # Fallback when floor object annotation is absent in the frame JSON.
    return float(gt_pose.position[2] - eye_height_m)


def camera_center_from_extrinsic(extrinsic_w2c: np.ndarray) -> np.ndarray:
    r = extrinsic_w2c[:3, :3]
    t = extrinsic_w2c[:3, 3]
    return (-r.T @ t).astype(np.float64)


def pixel_to_world_ray(
    u: float,
    v: float,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    pixel_h = np.array([u, v, 1.0], dtype=np.float64)
    ray_cam = np.linalg.inv(intrinsic) @ pixel_h
    ray_cam = normalize(ray_cam)
    if ray_cam is None:
        raise ValueError("Invalid camera ray from intrinsic inversion.")

    ray_world = normalize(extrinsic_w2c[:3, :3].T @ ray_cam)
    if ray_world is None:
        raise ValueError("Invalid world-space ray direction.")
    center = camera_center_from_extrinsic(extrinsic_w2c)
    return center, ray_world


def intersect_ray_with_z_plane(ray_origin: np.ndarray, ray_dir: np.ndarray, z_plane: float) -> np.ndarray:
    dz = float(ray_dir[2])
    if abs(dz) < 1e-9:
        raise ValueError("Ray is parallel to fixed Z plane; cannot intersect.")
    lam = (float(z_plane) - float(ray_origin[2])) / dz
    return ray_origin + lam * ray_dir


def pixel_to_world_on_z(
    u: float,
    v: float,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    z_plane: float,
) -> np.ndarray:
    origin, ray_dir = pixel_to_world_ray(u, v, intrinsic, extrinsic_w2c)
    return intersect_ray_with_z_plane(origin, ray_dir, z_plane)


def omega_to_world_direction(
    u: float,
    v: float,
    omega_deg: float,
    intrinsic: np.ndarray,
    extrinsic_w2c: np.ndarray,
    z_plane: float,
    step_px: float,
) -> Optional[np.ndarray]:
    rad = math.radians(float(omega_deg))
    du = math.cos(rad) * float(step_px)
    dv = -math.sin(rad) * float(step_px)
    p0 = pixel_to_world_on_z(u, v, intrinsic, extrinsic_w2c, z_plane)
    p1 = pixel_to_world_on_z(u + du, v + dv, intrinsic, extrinsic_w2c, z_plane)
    direction = p1 - p0
    direction[2] = 0.0
    return normalize(direction)


def compute_errors(pred: Pose3D, gt: Pose3D) -> Tuple[float, Optional[float]]:
    distance_m = float(np.linalg.norm(pred.position - gt.position))
    if pred.direction is None or gt.direction is None:
        return distance_m, None
    dot = float(np.clip(np.dot(pred.direction, gt.direction), -1.0, 1.0))
    angular_deg = float(math.degrees(math.acos(dot)))
    return distance_m, angular_deg


def single_point_metrics(
    distance_m: float,
    hit_radii: Sequence[float],
    mass_percentiles: Sequence[float],
) -> Tuple[Dict[float, float], Dict[float, float]]:
    hit_masses = {float(r): float(1.0 if distance_m <= float(r) else 0.0) for r in hit_radii}
    mass_radii = {float(p): float(distance_m) for p in mass_percentiles}
    return hit_masses, mass_radii


def compute_binary_recall(distance_error: float, recall_radii: Sequence[float]) -> Dict[float, float]:
    return {
        float(radius): float(1.0 if float(distance_error) <= float(radius) else 0.0)
        for radius in sorted(set(float(r) for r in recall_radii))
    }


def discover_mesh(scene_dir: Path, dataset: str = "3rscan") -> Path:
    if dataset == "scannet":
        sid = scene_dir.name
        for name in (f"{sid}_vh_clean_2.ply", f"{sid}_vh_clean_2.labels.ply"):
            path = scene_dir / name
            if path.exists():
                return path
    for name in PREFERRED_MESH_FILES:
        path = scene_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No known mesh file found in {scene_dir}")


def build_scene_iou_context(scene_dir: Path, dataset: str = "3rscan") -> SceneIoUContext:
    if o3d is None:
        raise RuntimeError("open3d is required to compute IoU metrics but is not installed.")

    mesh_path = discover_mesh(scene_dir, dataset=dataset)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty: {mesh_path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError(f"Mesh has no usable geometry: {mesh_path}")

    tri_points = vertices[triangles]
    edge_a = tri_points[:, 1] - tri_points[:, 0]
    edge_b = tri_points[:, 2] - tri_points[:, 0]
    tri_cross = np.cross(edge_a, edge_b)
    tri_areas = 0.5 * np.linalg.norm(tri_cross, axis=1)
    tri_centroids = tri_points.mean(axis=1)

    raycasting_scene = o3d.t.geometry.RaycastingScene()
    mesh_id = int(raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh)))
    return SceneIoUContext(
        raycasting_scene=raycasting_scene,
        mesh_id=mesh_id,
        tri_points=tri_points,
        tri_centroids=tri_centroids,
        tri_areas=tri_areas,
    )


def _camera_axes_from_forward(forward: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    fwd = np.asarray(forward, dtype=np.float64)
    norm = float(np.linalg.norm(fwd))
    if norm < 1e-6:
        return None
    fwd /= norm

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = np.cross(fwd, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-6:
        return None
    right /= right_norm

    up = np.cross(right, fwd)
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-6:
        return None
    up /= up_norm
    return fwd, right, up


def _visible_triangles_from_view(
    cam: np.ndarray,
    forward: Optional[np.ndarray],
    hfov: float,
    vfov: float,
    context: SceneIoUContext,
    near: float = 0.05,
    far: Optional[float] = None,
) -> set[int]:
    if forward is None:
        return set()
    axes = _camera_axes_from_forward(forward)
    if axes is None:
        return set()
    fwd_axis, right_axis, up_axis = axes

    cam = np.asarray(cam, dtype=np.float64)
    rel = context.tri_points - cam[None, None, :]
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

    selected_idx = np.nonzero(frustum_mask)[0]
    vectors = context.tri_centroids[selected_idx] - cam[None, :]
    distance = np.linalg.norm(vectors, axis=1)
    valid = distance > 1e-6
    if not np.any(valid):
        return set()

    selected_idx = selected_idx[valid]
    directions = vectors[valid] / distance[valid][:, None]
    rays = np.concatenate(
        [np.repeat(cam[None, :], len(selected_idx), axis=0), directions],
        axis=1,
    ).astype(np.float32)

    assert o3d is not None
    cast = context.raycasting_scene.cast_rays(o3d.core.Tensor(rays))
    primitive_ids = np.asarray(cast["primitive_ids"].numpy())
    geometry_ids = np.asarray(cast["geometry_ids"].numpy())
    hit_mask = (primitive_ids == selected_idx) & (geometry_ids == context.mesh_id)
    return {int(idx) for idx in selected_idx[hit_mask]}


def compute_view_iou_error(
    gt_cam: np.ndarray,
    gt_dir: Optional[np.ndarray],
    pred_cam: np.ndarray,
    pred_dir: Optional[np.ndarray],
    hfov_rad: float,
    vfov_rad: float,
    context: SceneIoUContext,
) -> Optional[float]:
    if gt_dir is None or pred_dir is None:
        return None

    gt_visible = _visible_triangles_from_view(gt_cam, gt_dir, hfov_rad, vfov_rad, context)
    pred_visible = _visible_triangles_from_view(pred_cam, pred_dir, hfov_rad, vfov_rad, context)
    if not gt_visible and not pred_visible:
        return None

    union = gt_visible | pred_visible
    if not union:
        return None
    intersection = gt_visible & pred_visible
    inter_area = float(context.tri_areas[list(intersection)].sum()) if intersection else 0.0
    union_area = float(context.tri_areas[list(union)].sum())
    if union_area <= 1e-9:
        return None
    iou = inter_area / union_area
    return float(1.0 - iou)


def visualize_debug_pose(
    scene_id: str,
    scene_dir: Path,
    gt_pose: Pose3D,
    pred_pose: Pose3D,
    dataset: str = "3rscan",
    marker_radius_m: float = 0.08,
    arrow_length_m: float = 0.8,
) -> None:
    # Lazy import to keep normal (non-visualize) runs on the fast path.
    import open3d as o3d

    mesh_path = discover_mesh(scene_dir, dataset=dataset)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    geoms: List[o3d.geometry.Geometry] = [mesh]

    def marker(center: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.TriangleMesh:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius_m)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(np.asarray(color, dtype=np.float64))
        sphere.translate(np.asarray(center, dtype=np.float64))
        return sphere

    def direction_line(
        center: np.ndarray,
        direction: Optional[np.ndarray],
        color: Tuple[float, float, float],
    ) -> Optional[o3d.geometry.LineSet]:
        if direction is None:
            return None
        d = normalize(np.asarray(direction, dtype=np.float64))
        if d is None:
            return None
        p0 = np.asarray(center, dtype=np.float64)
        p1 = p0 + d * float(arrow_length_m)
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(np.vstack([p0, p1]))
        line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
        line.colors = o3d.utility.Vector3dVector(np.asarray([color], dtype=np.float64))
        return line

    geoms.append(marker(gt_pose.position, (0.1, 0.8, 0.1)))
    geoms.append(marker(pred_pose.position, (0.9, 0.1, 0.1)))
    gt_line = direction_line(gt_pose.position, gt_pose.direction, (0.1, 0.8, 0.1))
    pred_line = direction_line(pred_pose.position, pred_pose.direction, (0.9, 0.1, 0.1))
    if gt_line is not None:
        geoms.append(gt_line)
    if pred_line is not None:
        geoms.append(pred_line)

    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"VLM Baseline Debug - {scene_id}",
        width=1400,
        height=900,
        mesh_show_back_face=True,
    )


def load_existing_results(path: Path) -> Tuple[List[dict], set[Tuple[str, str]]]:
    if not path.exists():
        return [], set()
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = payload.get("metrics", [])
    if not isinstance(payload, list):
        return [], set()
    completed = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        scene_id = str(item.get("scene_id", "")).strip()
        frame_id = str(item.get("frame_id", "")).strip()
        if scene_id and frame_id:
            completed.add((scene_id, frame_id))
    return payload, completed


def _to_float_metric_map(raw: object) -> Dict[float, float]:
    if not isinstance(raw, dict):
        return {}
    parsed: Dict[float, float] = {}
    for key, value in raw.items():
        try:
            parsed[float(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _normalize_radii(values: Iterable[float], name: str) -> List[float]:
    parsed: List[float] = []
    for raw in values:
        value = float(raw)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must contain finite values > 0, got {raw!r}.")
        parsed.append(value)
    return sorted(set(parsed))


def _metric_map_from_results(
    item: dict,
    map_key: str,
    fallback_distance_key: Optional[str] = None,
    fallback_radii: Optional[Sequence[float]] = None,
) -> Dict[float, float]:
    metric_map = _to_float_metric_map(item.get(map_key))
    if metric_map:
        return metric_map
    if fallback_distance_key is None or fallback_radii is None:
        return metric_map
    distance_value = item.get(fallback_distance_key)
    if distance_value is None:
        return metric_map
    try:
        distance = float(distance_value)
    except (TypeError, ValueError):
        return metric_map
    if not np.isfinite(distance):
        return metric_map
    if map_key == "recall":
        return compute_binary_recall(distance, fallback_radii)
    return metric_map


def aggregate_radius_metric_from_results(
    results: List[dict],
    radii: Sequence[float],
    map_key: str,
    fallback_distance_key: Optional[str] = None,
    fallback_radii: Optional[Sequence[float]] = None,
) -> Dict[float, Tuple[float, float]]:
    stats: Dict[float, Tuple[float, float]] = {}
    for radius in sorted(set(float(v) for v in radii)):
        values: List[float] = []
        for item in results:
            metric_map = _metric_map_from_results(
                item,
                map_key=map_key,
                fallback_distance_key=fallback_distance_key,
                fallback_radii=fallback_radii,
            )
            values.append(float(metric_map.get(radius, 0.0)))
        if values:
            stats[radius] = _aggregate(values)
    return stats


def build_metrics_table(
    results: List[dict],
    hit_radii: Sequence[float],
    mass_percentiles: Sequence[float],
    topk_k: int,
    recall_radii: Optional[Sequence[float]] = None,
) -> str:
    hit_radii_sorted = sorted(set(float(r) for r in hit_radii))
    recall_radii_sorted = [] if recall_radii is None else sorted(set(float(r) for r in recall_radii))
    mass_percentiles_sorted = sorted(set(float(p) for p in mass_percentiles))
    headers = [
        "Scene",
        "Frame",
        *[f"Hit@{r:.2f}m" for r in hit_radii_sorted],
        *[f"Recall@{r:.2f}m" for r in recall_radii_sorted],
        *[f"R{p:.0f}%" for p in mass_percentiles_sorted],
        f"TopK{topk_k} (m)",
        "Err (m)",
        "Ang err (deg)",
        "IoU err",
        "Matches",
        "Grid pts",
    ]

    rows: List[List[str]] = []
    for item in results:
        hit_map = _to_float_metric_map(item.get("hit_masses"))
        recall_map = _metric_map_from_results(
            item,
            map_key="recall",
            fallback_distance_key="distance_error",
            fallback_radii=recall_radii_sorted,
        )
        mass_map = _to_float_metric_map(item.get("mass_radii"))
        ang_err = item.get("angular_error_deg")
        iou_err = item.get("iou_error")
        rows.append([
            str(item.get("scene_id", "")),
            str(item.get("frame_id", "")),
            *[f"{hit_map.get(r, 0.0):.3f}" for r in hit_radii_sorted],
            *[f"{recall_map.get(r, 0.0):.3f}" for r in recall_radii_sorted],
            *[
                f"{mass_map.get(p, float('nan')):.3f}"
                if np.isfinite(mass_map.get(p, float("nan"))) else "-"
                for p in mass_percentiles_sorted
            ],
            f"{float(item.get('topk_min_dist', float('nan'))):.3f}",
            f"{float(item.get('distance_error', float('nan'))):.3f}",
            "-" if ang_err is None else f"{float(ang_err):.2f}",
            "-" if iou_err is None else f"{float(iou_err):.3f}",
            str(int(item.get("matched_objects", 0))),
            str(int(item.get("grid_points", 0))),
        ])

    if not rows:
        return ""

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in col_widths)
    lines = [fmt_row(headers), separator]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def _format_failures_section(failures: List[FailureRecord]) -> str:
    lines = [
        "Failed frames summary",
        "--------------------",
        f"Total failed frames: {len(failures)}",
    ]
    if not failures:
        lines.append("No failed frames.")
        return "\n".join(lines)

    for fail in failures:
        lines.append(f"{fail.scene_id} | {fail.frame_id} | {fail.reason}")
    return "\n".join(lines)


def _aggregate(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(np.median(arr))


def compute_aggregate_metrics(
    results: List[dict],
    args: argparse.Namespace,
) -> Tuple[dict, List[str]]:
    hit_stats = aggregate_radius_metric_from_results(results, args.hit_radii, map_key="hit_masses")
    recall_stats = aggregate_radius_metric_from_results(
        results,
        args.recall_radii,
        map_key="recall",
        fallback_distance_key="distance_error",
        fallback_radii=args.recall_radii,
    )

    mass_radius_stats: Dict[float, Tuple[float, float]] = {}
    for p in sorted(set(float(x) for x in args.mass_percentiles)):
        vals = []
        for item in results:
            value = _to_float_metric_map(item.get("mass_radii")).get(p, float("nan"))
            if np.isfinite(value):
                vals.append(value)
        if vals:
            mass_radius_stats[p] = _aggregate(vals)

    mean_topk, med_topk = _aggregate([float(item.get("topk_min_dist", float("nan"))) for item in results])
    mean_err, med_err = _aggregate([float(item.get("distance_error", float("nan"))) for item in results])

    ang_values = [
        float(item["angular_error_deg"])
        for item in results
        if item.get("angular_error_deg") is not None
    ]
    mean_ang: Optional[float] = None
    med_ang: Optional[float] = None
    if ang_values:
        mean_ang, med_ang = _aggregate(ang_values)

    iou_values = [
        float(item["iou_error"])
        for item in results
        if item.get("iou_error") is not None
    ]
    mean_iou: Optional[float] = None
    med_iou: Optional[float] = None
    if iou_values:
        mean_iou, med_iou = _aggregate(iou_values)

    agg_lines = [
        "Aggregate metrics ---------------------------------------",
        f"  TopK{args.top_k_min_dist} min dist (m): mean={mean_topk:.3f} | median={med_topk:.3f}",
        f"  Distance error (m)      : mean={mean_err:.3f} | median={med_err:.3f}",
        "---------------------------------------------------------",
    ]
    for r in sorted(hit_stats):
        mean_hit, med_hit = hit_stats[r]
        agg_lines.insert(-1, f"  Hit@{r:.2f}m              : mean={mean_hit:.3f} | median={med_hit:.3f}")
    for r in sorted(recall_stats):
        mean_recall, med_recall = recall_stats[r]
        agg_lines.insert(-1, f"  Recall@{r:.2f}m           : mean={mean_recall:.3f} | median={med_recall:.3f}")
    for p in sorted(mass_radius_stats):
        mean_r, med_r = mass_radius_stats[p]
        agg_lines.insert(-1, f"  Mass-radius R{p:.0f}% (m): mean={mean_r:.3f} | median={med_r:.3f}")
    if mean_ang is not None and med_ang is not None:
        agg_lines.insert(-1, f"  Angular error (deg)   : mean={mean_ang:.2f} | median={med_ang:.2f}")
    if mean_iou is not None and med_iou is not None:
        agg_lines.insert(-1, f"  View IoU error        : mean={mean_iou:.3f} | median={med_iou:.3f}")

    aggregate = {
        "hit_masses": {
            str(r): {"mean": mean_hit, "median": med_hit}
            for r, (mean_hit, med_hit) in hit_stats.items()
        },
        "recall": {
            str(r): {"mean": mean_recall, "median": med_recall}
            for r, (mean_recall, med_recall) in recall_stats.items()
        },
        "mass_radii": {
            str(p): {"mean": mean_r, "median": med_r}
            for p, (mean_r, med_r) in mass_radius_stats.items()
        },
        "topk_min_dist": {"mean": mean_topk, "median": med_topk},
        "distance_error": {"mean": mean_err, "median": med_err},
        "angular_error_deg": None if mean_ang is None else {"mean": mean_ang, "median": med_ang},
        "iou_error": None if mean_iou is None else {"mean": mean_iou, "median": med_iou},
        "hit_radii": [float(r) for r in args.hit_radii],
        "recall_radii": [float(r) for r in args.recall_radii],
        "mass_percentiles": [float(p) for p in args.mass_percentiles],
        "top_k_min_dist": int(args.top_k_min_dist),
        "h_fov_deg": float(args.h_fov_deg),
        "v_fov_deg": float(args.v_fov_deg),
    }
    return aggregate, agg_lines


def _select_frame_paths(
    frame_paths: List[Path],
    policy: str,
    frame_index: int,
    rng: np.random.Generator,
) -> List[Path]:
    """Apply a frame selection policy and return the selected path(s)."""
    if not frame_paths:
        return []

    def _load(fp: Path) -> dict:
        return json.loads(fp.read_text())

    def _visible_count(fp: Path) -> int:
        frame = _load(fp)
        objs = frame.get("visible_objects", {}) or {}
        return len(objs)

    def _total_pixels(fp: Path) -> int:
        frame = _load(fp)
        objs = frame.get("visible_objects", {}) or {}
        total = 0
        for obj in objs.values():
            try:
                total += int((obj or {}).get("pixel_count", 0))
            except (TypeError, ValueError):
                continue
        return total

    if policy == "first":
        return [frame_paths[0]]
    if policy == "index":
        return [frame_paths[frame_index % len(frame_paths)]]
    if policy == "random":
        return [frame_paths[int(rng.integers(0, len(frame_paths)))]]
    if policy == "max_visible":
        best = min(
            frame_paths,
            key=lambda fp: (-_visible_count(fp), -_total_pixels(fp), fp.name),
        )
        return [best]
    if policy == "max_pixels":
        return [max(frame_paths, key=_total_pixels)]

    raise ValueError(f"Unknown frame selection policy '{policy}'")


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    args.hit_radii = _normalize_radii(args.hit_radii, "hit_radii")
    args.recall_radii = _normalize_radii(args.recall_radii, "recall_radii")
    args.mass_percentiles = [float(p) for p in args.mass_percentiles]
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)

    if args.query_root is None:
        args.query_root = args.root

    start_time = time.time()
    hfov_rad = math.radians(args.h_fov_deg)
    vfov_rad = math.radians(args.v_fov_deg)
    scene_dirs = find_scene_dirs(args.root, args.scene_ids, args.max_scenes)
    if not scene_dirs:
        print(f"No scene directories found under {args.root}")
        return

    existing_results: List[dict] = []
    completed: set[Tuple[str, str]] = set()
    if args.resume:
        existing_results, completed = load_existing_results(args.save_metrics)
        print(f"Resume enabled: loaded {len(existing_results)} existing records.")

    print(f"Loading model: {args.model_id}")
    bundle = load_qwen_bundle(args)

    results: List[dict] = list(existing_results)
    failures: List[FailureRecord] = []
    processed_new = 0

    skipped_scenes = 0
    for s_idx, scene_dir in enumerate(scene_dirs, start=1):
        scene_id = scene_dir.name

        # Check for pending frames before doing any heavy I/O.
        frame_paths = frame_json_paths(scene_dir, args.max_frames_per_scene, query_root=args.query_root)
        if not frame_paths:
            failures.append(FailureRecord(scene_id=scene_id, frame_id="*", reason="No frame-*.json files found"))
            continue

        # Apply frame selection policy
        if args.frame_policy != "all":
            frame_paths = _select_frame_paths(frame_paths, args.frame_policy, args.frame_index, rng)

        if completed:
            has_pending = False
            for fp in frame_paths:
                fid = str(json.loads(fp.read_text()).get("image_index", fp.stem))
                if (scene_id, fid) not in completed:
                    has_pending = True
                    break
            if not has_pending:
                skipped_scenes += 1
                continue

        print(f"[{s_idx:04d}/{len(scene_dirs):04d}] Scene {scene_id} (skipped {skipped_scenes} completed scenes)")

        try:
            topdown_image, camera_npz = resolve_topdown_paths(scene_dir, query_root=args.query_root)
            intrinsic, extrinsic = load_topdown_camera(camera_npz)
        except Exception as exc:  # noqa: BLE001
            failures.append(FailureRecord(scene_id=scene_id, frame_id="*", reason=str(exc)))
            print(f"  [WARN] Scene skipped: {exc}")
            continue

        scene_iou_context: Optional[SceneIoUContext] = None
        try:
            scene_iou_context = build_scene_iou_context(scene_dir, dataset=args.dataset)
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] IoU disabled for scene {scene_id}: {exc}")

        for f_idx, frame_path in enumerate(frame_paths, start=1):
            try:
                frame = json.loads(frame_path.read_text())
                frame_id = str(frame.get("image_index", frame_path.stem))
                key = (scene_id, frame_id)
                if key in completed:
                    continue

                description = str(frame.get("description", "")).strip()
                if not description:
                    raise ValueError("Missing non-empty 'description' in frame JSON.")

                gt_pose = gt_pose_from_frame(frame, frame_path)
                floor_z = estimate_floor_z(frame, gt_pose, args.eye_height_m)
                pred_z = floor_z + float(args.eye_height_m)

                last_err: Optional[Exception] = None
                for _attempt in range(args.retry_count + 1):
                    try:
                        pred_u, pred_v, pred_w, raw = run_qwen_inference(
                            bundle=bundle,
                            image_path=topdown_image,
                            description=description,
                            max_new_tokens=args.max_new_tokens,
                        )
                        pred_pos = pixel_to_world_on_z(pred_u, pred_v, intrinsic, extrinsic, pred_z)
                        pred_dir = omega_to_world_direction(
                            u=pred_u,
                            v=pred_v,
                            omega_deg=pred_w,
                            intrinsic=intrinsic,
                            extrinsic_w2c=extrinsic,
                            z_plane=pred_z,
                            step_px=args.direction_step_px,
                        )
                        pred_pose = Pose3D(position=pred_pos, direction=pred_dir)
                        dist_m, ang_deg = compute_errors(pred_pose, gt_pose)
                        hit_masses, mass_radii = single_point_metrics(
                            distance_m=dist_m,
                            hit_radii=args.hit_radii,
                            mass_percentiles=args.mass_percentiles,
                        )
                        recall = compute_binary_recall(dist_m, args.recall_radii)
                        iou_error = None
                        if scene_iou_context is not None:
                            iou_error = compute_view_iou_error(
                                gt_cam=gt_pose.position,
                                gt_dir=gt_pose.direction,
                                pred_cam=pred_pose.position,
                                pred_dir=pred_pose.direction,
                                hfov_rad=hfov_rad,
                                vfov_rad=vfov_rad,
                                context=scene_iou_context,
                            )
                        entry = {
                            "scene_id": scene_id,
                            "frame_id": frame_id,
                            "hit_masses": {str(k): v for k, v in hit_masses.items()},
                            "recall": {str(k): v for k, v in recall.items()},
                            "mass_radii": {str(k): v for k, v in mass_radii.items()},
                            "topk_min_dist": float(dist_m),
                            "distance_error": float(dist_m),
                            "angular_error_deg": ang_deg,
                            "grid_points": 1,
                            "matched_objects": 0,
                            "iou_error": iou_error,
                            "pred_u": int(pred_u),
                            "pred_v": int(pred_v),
                            "pred_w_deg": float(pred_w),
                            "pred_position_xyz": [float(x) for x in pred_pose.position.tolist()],
                            "pred_direction_xyz": None if pred_pose.direction is None else [float(x) for x in pred_pose.direction.tolist()],
                            "gt_position_xyz": [float(x) for x in gt_pose.position.tolist()],
                            "gt_direction_xyz": None if gt_pose.direction is None else [float(x) for x in gt_pose.direction.tolist()],
                            "raw_model_response": raw,
                        }
                        results.append(entry)
                        completed.add(key)
                        processed_new += 1
                        if args.visualize:
                            visualize_debug_pose(scene_id=scene_id, scene_dir=scene_dir, gt_pose=gt_pose, pred_pose=pred_pose, dataset=args.dataset)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                else:
                    raise RuntimeError(str(last_err) if last_err is not None else "Unknown inference error.")

                if f_idx % 10 == 0:
                    print(f"  processed {f_idx}/{len(frame_paths)} frames")
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    FailureRecord(
                        scene_id=scene_id,
                        frame_id=str(frame_path.stem),
                        reason=str(exc),
                    )
                )

        # Save incrementally for long runs.
        save_json(args.save_metrics, results)

    elapsed = time.time() - start_time
    params_text = format_args_section(args)
    failure_text = _format_failures_section(failures)

    if not results:
        no_metrics_log = "\n\n".join([
            "No scenes produced metrics.",
            params_text,
            failure_text,
        ]).rstrip() + "\n"
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text(no_metrics_log)
        save_json(args.save_metrics, {
            "metrics": [],
            "failures": [{"scene_id": f.scene_id, "frame_id": f.frame_id, "reason": f.reason} for f in failures],
            "aggregate": None,
        })
        print(f"Saved metrics to {args.save_metrics}")
        print(f"Saved log to {args.log_file}")
        print(f"Processed new frames: {processed_new}")
        print(f"Failures: {len(failures)}")
        return

    table_text = build_metrics_table(
        results=results,
        hit_radii=args.hit_radii,
        mass_percentiles=args.mass_percentiles,
        topk_k=args.top_k_min_dist,
        recall_radii=args.recall_radii,
    )
    aggregate, aggregate_lines = compute_aggregate_metrics(results, args)
    aggregate_lines = list(aggregate_lines)
    aggregate_lines.insert(-1, f"  Processed frames        : {len(results)}")
    aggregate_lines.insert(-1, f"  Failed frames           : {len(failures)}")
    aggregate_lines.insert(-1, f"  Wall time (s)           : {elapsed:.2f}")

    log_sections: List[str] = [params_text]
    if table_text:
        log_sections.extend(["Scene/frame summary table", table_text])
    log_sections.append("\n".join(aggregate_lines))
    log_sections.append(failure_text)
    log_payload = "\n\n".join(log_sections).rstrip() + "\n"
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.write_text(log_payload)

    metrics_payload = {
        "metrics": results,
        "failures": [
            {"scene_id": fail.scene_id, "frame_id": fail.frame_id, "reason": fail.reason}
            for fail in failures
        ],
        "aggregate": aggregate,
    }
    save_json(args.save_metrics, metrics_payload)

    print(f"Saved metrics to {args.save_metrics}")
    print(f"Saved log to {args.log_file}")
    print(f"Processed new frames: {processed_new}")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()
