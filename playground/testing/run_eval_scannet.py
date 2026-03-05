#!/usr/bin/env python3
"""
Evaluate predicted pose against GT for a ScanNet scene and visualize both on the mesh.
"""

import json
import math
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required: pip install open3d")

from eval_pose_iou_ang import (
    normalize,
    compute_view_iou,
    _camera_axes_from_forward,
)


def build_iou_context_scannet(scene_dir, mesh_path):
    """Build raycasting context from a specific mesh path."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    if mesh.is_empty():
        raise ValueError(f"Mesh is empty: {mesh_path}")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    tri_points = verts[tris]
    edge_a = tri_points[:, 1] - tri_points[:, 0]
    edge_b = tri_points[:, 2] - tri_points[:, 0]
    tri_cross = np.cross(edge_a, edge_b)
    tri_areas = 0.5 * np.linalg.norm(tri_cross, axis=1)
    tri_centroids = tri_points.mean(axis=1)
    raycasting_scene = o3d.t.geometry.RaycastingScene()
    mesh_id = int(raycasting_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh)))
    return raycasting_scene, mesh_id, tri_points, tri_centroids, tri_areas

# --- Config ---
DATASET_ROOT = Path("/media/klrshak/Backup/Datasets/scannet_scenes_100/scans")
SCENE_ID = "scene0000_01"
FRAME_ID = "001269"
PRED_POS = np.array([5.036613941192627, 4.492875099182129, 1.5716819763183594])
PRED_DIR = np.array([0.9696859820286318, -0.24357000244776697, 0.01956400175738604])
HFOV_DEG = 39.31
VFOV_DEG = 64.76
NEAR = 0.05
FAR = None

# --- Load GT ---
scene_dir = DATASET_ROOT / SCENE_ID
frame_path = scene_dir / "output" / "descriptions" / f"{FRAME_ID}.json"
frame = json.loads(frame_path.read_text())
pose = np.asarray(frame["scene_pose"], dtype=np.float64)
gt_pos = pose[:3, 3].copy()
gt_dir = normalize(pose[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
pred_dir_norm = normalize(PRED_DIR)

print(f"Scene: {SCENE_ID}, Frame: {FRAME_ID}")
print(f"GT  position : {gt_pos.tolist()}")
print(f"GT  direction: {gt_dir.tolist()}")
print(f"Pred position: {PRED_POS.tolist()}")
print(f"Pred direction: {pred_dir_norm.tolist()}")

# --- Angular error ---
dot = float(np.clip(np.dot(pred_dir_norm, gt_dir), -1.0, 1.0))
angular_error_deg = math.degrees(math.acos(dot))
print(f"\nAngular error: {angular_error_deg:.4f} deg")

# --- Position error ---
pos_error = float(np.linalg.norm(PRED_POS - gt_pos))
print(f"Position error: {pos_error:.4f} m")

# --- IoU ---
# Find the mesh
mesh_candidates = [
    scene_dir / f"{SCENE_ID}_vh_clean_2.ply",
    scene_dir / f"{SCENE_ID}_vh_clean_2.labels.ply",
]
mesh_path = None
for p in mesh_candidates:
    if p.exists():
        mesh_path = p
        break
if mesh_path is None:
    raise FileNotFoundError(f"No mesh found in {scene_dir}")

print(f"\nUsing mesh: {mesh_path}")
raycasting_scene, mesh_id, tri_points, tri_centroids, tri_areas = build_iou_context_scannet(scene_dir, mesh_path)
iou = compute_view_iou(
    gt_cam=gt_pos,
    gt_dir=gt_dir,
    pred_cam=PRED_POS,
    pred_dir=pred_dir_norm,
    hfov_rad=math.radians(HFOV_DEG),
    vfov_rad=math.radians(VFOV_DEG),
    raycasting_scene=raycasting_scene,
    mesh_id=mesh_id,
    tri_points=tri_points,
    tri_centroids=tri_centroids,
    tri_areas=tri_areas,
    near=NEAR,
    far=FAR,
)
if iou is not None:
    print(f"View IoU: {iou:.6f}")
    print(f"IoU error: {1.0 - iou:.6f}")
else:
    print("View IoU: n/a")

# --- Visualization ---
print(f"\nLoading mesh for visualization: {mesh_path}")
mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
mesh.compute_vertex_normals()

def make_camera_frustum(pos, forward, color, scale=0.3):
    """Create a simple camera frustum visualization."""
    axes = _camera_axes_from_forward(forward)
    if axes is None:
        return []
    fwd, right, up = axes

    # Frustum corners
    hw = scale * math.tan(math.radians(HFOV_DEG) / 2)
    hh = scale * math.tan(math.radians(VFOV_DEG) / 2)

    corners = [
        pos + scale * fwd + hw * right + hh * up,
        pos + scale * fwd - hw * right + hh * up,
        pos + scale * fwd - hw * right - hh * up,
        pos + scale * fwd + hw * right - hh * up,
    ]

    lines = []
    for c in corners:
        lines.append([pos, c])
    for i in range(4):
        lines.append([corners[i], corners[(i + 1) % 4]])

    points = [pos] + corners
    line_indices = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(line_indices))

    # Camera position sphere
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    sphere.translate(pos)
    sphere.paint_uniform_color(color)

    # Direction arrow
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.015, cone_radius=0.03,
        cylinder_height=scale * 0.7, cone_height=scale * 0.3,
    )
    # Rotate arrow to align with forward direction
    z_axis = np.array([0, 0, 1], dtype=np.float64)
    rotation_axis = np.cross(z_axis, fwd)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm > 1e-6:
        rotation_axis /= rotation_axis_norm
        angle = math.acos(np.clip(np.dot(z_axis, fwd), -1, 1))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
        arrow.rotate(R, center=np.array([0, 0, 0]))
    elif np.dot(z_axis, fwd) < 0:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * math.pi)
        arrow.rotate(R, center=np.array([0, 0, 0]))
    arrow.translate(pos)
    arrow.paint_uniform_color(color)

    return [line_set, sphere, arrow]

gt_color = [0.0, 1.0, 0.0]   # green
pred_color = [1.0, 0.0, 0.0]  # red

gt_geoms = make_camera_frustum(gt_pos, gt_dir, gt_color, scale=0.5)
pred_geoms = make_camera_frustum(PRED_POS, pred_dir_norm, pred_color, scale=0.5)

print("\nVisualization: Green = GT, Red = Predicted")
print("Close the window to exit.")
o3d.visualization.draw_geometries(
    [mesh] + gt_geoms + pred_geoms,
    window_name=f"GT (green) vs Pred (red) | Ang err: {angular_error_deg:.1f}° | IoU: {iou:.4f}" if iou else f"GT vs Pred | Ang err: {angular_error_deg:.1f}°",
    width=1280,
    height=720,
)
