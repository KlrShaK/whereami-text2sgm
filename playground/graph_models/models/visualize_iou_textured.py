#!/usr/bin/env python3
"""
visualize_iou_textured.py
-------------------------
Standalone IoU visualization tool that loads a candidates JSON (from mk5) and
renders the textured mesh with GT/predicted frustums and IoU triangle overlays.

Improvements over the inline mk5 IoU visualizer:
  - Loads the actual textured mesh (OBJ with textures baked to vertex colors)
    instead of flat-shaded or transparency-only meshes.
  - IoU overlap regions are tinted on top of the real texture.
  - GT and predicted frustums are clearly labelled with 3D text labels.
  - Camera markers (spheres) are labelled and colour-coded.

Usage:
    python visualize_iou_textured.py \
        --candidates eval/eval_candidates_mk5_weighted_scannet_abu_final.json \
        --root /path/to/scannet_scenes \
        --dataset scannet \
        --scene_id scene0000_00 \
        [--frame_id 001541]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).resolve().parent / ".." / "data_processing"))
sys.path.append(str(Path(__file__).resolve().parent / ".." / ".." / ".."))

# --------------------------------------------------------------------------- #
# Mesh loading with texture support                                           #
# --------------------------------------------------------------------------- #

def _find_textured_mesh_path(scene_dir: Path, dataset: str) -> Optional[Path]:
    """Find the best mesh file that may have textures."""
    sid = scene_dir.name
    if dataset == "3rscan":
        # Keep the same discovery order used by topdown_3rscan.py.
        for name in [
            "mesh.refined.v2.obj",
            "labels.instances.annotated.v2.ply",
            "mesh.refined.ply",
            "mesh.refined.obj",
            "mesh.obj",
        ]:
            path = scene_dir / name
            if path.exists():
                return path
    elif dataset == "scannet":
        for name in [f"{sid}_vh_clean_2.ply", f"{sid}_vh_clean_2.labels.ply"]:
            p = scene_dir / name
            if p.exists():
                return p
    return None


def _find_texture_image(mesh_path: Path,
                        mesh: o3d.geometry.TriangleMesh) -> Optional[np.ndarray]:
    """Locate and return the texture image as (H, W, 3) uint8, or None."""
    if mesh.textures:
        return np.asarray(mesh.textures[0])

    from PIL import Image

    mesh_dir = mesh_path.parent
    stem = mesh_path.stem
    base = stem.split(".")[0] + "." + stem.split(".")[1] if "." in stem else stem
    for candidate in [
        mesh_dir / f"{base}_0.png",
        mesh_dir / f"{stem}_0.png",
        mesh_dir / f"{stem}.png",
        mesh_dir / "texture.png",
    ]:
        if candidate.exists():
            return np.array(Image.open(candidate).convert("RGB"))

    mtl_path = mesh_path.with_suffix(".mtl")
    if not mtl_path.exists():
        mtl_path = mesh_dir / (base + ".mtl")
    if mtl_path.exists():
        for line in mtl_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("map_Kd"):
                tex_file = mesh_dir / line.split(None, 1)[1]
                if tex_file.exists():
                    return np.array(Image.open(tex_file).convert("RGB"))
    return None


def _bake_texture_to_vertex_colors(mesh_path: Path,
                                   mesh: o3d.geometry.TriangleMesh) -> bool:
    """Bake UV-mapped texture into vertex colors in-place. Returns True on success."""
    if not mesh.has_triangle_uvs():
        return False

    triangle_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    num_verts = len(mesh.vertices)

    if triangle_uvs.size == 0 or faces.size == 0:
        return False

    tex_img = _find_texture_image(mesh_path, mesh)
    if tex_img is None:
        return False

    h, w = tex_img.shape[:2]
    uvs_per_face = triangle_uvs.reshape(-1, 3, 2)

    all_vert_ids = faces.reshape(-1)
    all_uvs = uvs_per_face.reshape(-1, 2).astype(np.float64)

    uv_sum_u = np.bincount(all_vert_ids, weights=all_uvs[:, 0], minlength=num_verts)
    uv_sum_v = np.bincount(all_vert_ids, weights=all_uvs[:, 1], minlength=num_verts)
    uv_count = np.bincount(all_vert_ids, minlength=num_verts).astype(np.float64)

    valid = uv_count > 0
    uv_avg = np.zeros((num_verts, 2), dtype=np.float32)
    uv_avg[valid, 0] = (uv_sum_u[valid] / uv_count[valid]).astype(np.float32)
    uv_avg[valid, 1] = (uv_sum_v[valid] / uv_count[valid]).astype(np.float32)

    px = np.clip((uv_avg[:, 0] * w).astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv_avg[:, 1]) * h).astype(int), 0, h - 1)

    vertex_colors = tex_img[py, px, :3].astype(np.float64) / 255.0
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return True


def _has_valid_vertex_colors(colors: np.ndarray, n_verts: int) -> bool:
    if colors.shape[0] != n_verts or colors.size == 0:
        return False
    if np.allclose(colors, 0.0):
        return False
    return True


def load_textured_mesh(scene_dir: Path,
                       dataset: str) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
    """Load mesh using the same two-source pattern as topdown_3rscan.py.

    Returns:
        display_mesh: mesh used for base rendering (keeps UV textures for 3RScan).
        tint_mesh: mesh with reliable vertex colors for IoU tint overlays.
    """
    mesh_path = _find_textured_mesh_path(scene_dir, dataset)
    if mesh_path is None:
        raise FileNotFoundError(f"No mesh found in {scene_dir} for dataset={dataset}")

    display_mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    display_mesh.compute_vertex_normals()
    tint_mesh = o3d.geometry.TriangleMesh(display_mesh)
    tint_mesh.compute_vertex_normals()

    tint_colors = np.asarray(tint_mesh.vertex_colors, dtype=np.float32)
    n_verts = len(tint_mesh.vertices)
    if not _has_valid_vertex_colors(tint_colors, n_verts):
        _bake_texture_to_vertex_colors(mesh_path, tint_mesh)

    tint_colors = np.asarray(tint_mesh.vertex_colors, dtype=np.float32)
    if not _has_valid_vertex_colors(tint_colors, n_verts):
        tint_mesh.paint_uniform_color([0.75, 0.75, 0.75])

    # For ScanNet, base rendering should use vertex colors too.
    if dataset == "scannet":
        display_colors = np.asarray(display_mesh.vertex_colors, dtype=np.float32)
        if not _has_valid_vertex_colors(display_colors, len(display_mesh.vertices)):
            display_mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.asarray(tint_mesh.vertex_colors)
            )

    return display_mesh, tint_mesh


def _remove_roof(mesh: o3d.geometry.TriangleMesh,
                 z_percentile: float = 90.0) -> o3d.geometry.TriangleMesh:
    """Remove triangles whose centroid Z is above z_percentile of all vertex Z values.

    Returns a new mesh with roof triangles stripped out while preserving
    available texture data (UVs/textures or vertex colors).
    """
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if tris.size == 0:
        return mesh

    tri_pts = verts[tris]
    tri_z = tri_pts[:, :, 2].mean(axis=1)
    z_cutoff = np.percentile(verts[:, 2], z_percentile)

    keep_mask = tri_z < z_cutoff
    kept_tris = tris[keep_mask]

    if kept_tris.size == 0:
        return mesh

    uniq, inv = np.unique(kept_tris.reshape(-1), return_inverse=True)
    new_verts = verts[uniq]
    new_tris = inv.reshape(-1, 3)

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_verts)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_tris)

    # Preserve UV texture mapping for 3RScan-style textured OBJ meshes.
    tri_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32)
    if mesh.has_triangle_uvs() and tri_uvs.shape == (len(tris) * 3, 2):
        kept_uvs = tri_uvs.reshape(-1, 3, 2)[keep_mask].reshape(-1, 2)
        new_mesh.triangle_uvs = o3d.utility.Vector2dVector(kept_uvs.astype(np.float64))
        if len(mesh.textures) > 0:
            new_mesh.textures = mesh.textures
        tri_mats = np.asarray(mesh.triangle_material_ids, dtype=np.int32)
        if tri_mats.size == len(tris):
            new_mesh.triangle_material_ids = o3d.utility.IntVector(
                tri_mats[keep_mask].tolist()
            )

    has_colors = (np.asarray(mesh.vertex_colors).shape[0] == len(verts))
    if has_colors:
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.asarray(mesh.vertex_colors)[uniq]
        )

    new_mesh.compute_vertex_normals()
    return new_mesh


# --------------------------------------------------------------------------- #
# Frustum construction (standalone, imported pattern from mk4)                #
# --------------------------------------------------------------------------- #

def create_camera_frustum(center: np.ndarray,
                          forward: Optional[np.ndarray],
                          colour: Tuple[float, float, float],
                          h_fov: float,
                          v_fov: float,
                          scale: float = 0.6) -> Optional[o3d.geometry.LineSet]:
    """Return a line-based camera frustum visualisation."""
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
    far_centre = centre + fwd * depth
    corners = [
        far_centre + right * half_w + up * half_h,
        far_centre - right * half_w + up * half_h,
        far_centre - right * half_w - up * half_h,
        far_centre + right * half_w - up * half_h,
    ]

    points = [centre] + corners
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1],
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(points))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines))
    ls.paint_uniform_color(colour)
    return ls


# --------------------------------------------------------------------------- #
# IoU triangle visibility computation                                         #
# --------------------------------------------------------------------------- #

def _visible_triangles_from_view(
    cam: np.ndarray,
    direction: Optional[np.ndarray],
    h_fov: float,
    v_fov: float,
    rc: o3d.t.geometry.RaycastingScene,
    geom_id: int,
    tri_centroids: np.ndarray,
    near: float = 0.05,
    far: Optional[float] = None,
) -> Set[int]:
    """Return set of triangle indices visible from cam looking in direction."""
    if direction is None:
        return set()

    fwd = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        return set()
    fwd /= norm

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, up))) > 0.95:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)

    half_h_fov = h_fov / 2.0
    half_v_fov = v_fov / 2.0

    cam_pos = np.asarray(cam, dtype=np.float64)
    diff = tri_centroids - cam_pos[None, :]
    dists = np.linalg.norm(diff, axis=1)

    mask = dists > near
    if far is not None:
        mask &= dists < far

    diff_normed = diff.copy()
    diff_normed[dists > 0] /= dists[dists > 0, None]

    dot_fwd = diff_normed @ fwd
    dot_right = diff_normed @ right
    dot_up = diff_normed @ up

    safe_fwd = np.clip(dot_fwd, 1e-9, None)
    angle_h = np.abs(np.arctan2(dot_right, safe_fwd))
    angle_v = np.abs(np.arctan2(dot_up, safe_fwd))

    in_fov = mask & (dot_fwd > 0) & (angle_h < half_h_fov) & (angle_v < half_v_fov)

    candidate_indices = np.where(in_fov)[0]
    if len(candidate_indices) == 0:
        return set()

    origins = np.broadcast_to(cam_pos, (len(candidate_indices), 3)).astype(np.float32)
    dirs_np = diff[candidate_indices].astype(np.float32)
    norms = np.linalg.norm(dirs_np, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    dirs_np /= norms

    rays = np.concatenate([origins, dirs_np], axis=1)
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    result = rc.cast_rays(rays_t)

    hit_geom = result["geometry_ids"].numpy().astype(np.int64)
    hit_tri = result["primitive_ids"].numpy().astype(np.int64)
    t_hit = result["t_hit"].numpy()

    visible = set()
    for i, ci in enumerate(candidate_indices):
        if hit_geom[i] != geom_id:
            continue
        if not np.isfinite(t_hit[i]):
            continue
        if hit_tri[i] == ci:
            visible.add(int(ci))

    return visible


# --------------------------------------------------------------------------- #
# Subset mesh builder for IoU overlay                                         #
# --------------------------------------------------------------------------- #

def _build_tinted_overlay(base_mesh: o3d.geometry.TriangleMesh,
                          tri_indices: Set[int],
                          tint_colour: np.ndarray,
                          tint_strength: float = 0.45) -> Optional[o3d.geometry.TriangleMesh]:
    """Extract triangles and tint existing vertex colours toward tint_colour."""
    if not tri_indices:
        return None

    idx_arr = np.asarray(sorted(tri_indices), dtype=np.int64)
    verts_arr = np.asarray(base_mesh.vertices)
    tris_arr = np.asarray(base_mesh.triangles)[idx_arr]

    uniq, inv = np.unique(tris_arr.reshape(-1), return_inverse=True)
    new_verts = verts_arr[uniq]
    new_tris = inv.reshape(-1, 3)

    base_colors = np.asarray(base_mesh.vertex_colors)
    if base_colors.shape[0] == verts_arr.shape[0]:
        orig_colors = base_colors[uniq]
    else:
        orig_colors = np.full((len(uniq), 3), 0.7)

    tinted = np.clip(
        (1.0 - tint_strength) * orig_colors + tint_strength * tint_colour,
        0.0, 1.0,
    )

    sub = o3d.geometry.TriangleMesh()
    sub.vertices = o3d.utility.Vector3dVector(new_verts)
    sub.triangles = o3d.utility.Vector3iVector(new_tris)
    sub.vertex_colors = o3d.utility.Vector3dVector(tinted)
    sub.compute_vertex_normals()
    return sub


# --------------------------------------------------------------------------- #
# Main visualizer                                                             #
# --------------------------------------------------------------------------- #

def visualize_scene_iou(scene_dir: Path,
                        dataset: str,
                        gt_position: np.ndarray,
                        gt_direction: Optional[np.ndarray],
                        pred_position: np.ndarray,
                        pred_direction: Optional[np.ndarray],
                        scene_id: str = "",
                        frame_id: str = "",
                        h_fov_deg: float = 39.31,
                        v_fov_deg: float = 64.76,
                        frustum_scale: float = 0.8,
                        gt_sphere_radius: float = 0.10,
                        pred_sphere_radius: float = 0.09,
                        tint_strength: float = 0.45,
                        show_base_textured: bool = True,
                        remove_roof: bool = True,
                        roof_z_percentile: float = 90.0,
                        description: str = "",
                        desc_wrap_width: int = 50,
                        show_legend: bool = True) -> None:
    """Launch the Open3D visualizer with textured IoU overlay."""
    hfov_rad = math.radians(h_fov_deg)
    vfov_rad = math.radians(v_fov_deg)

    # Load both rendering and tint sources.
    # 3RScan: base mesh keeps UV textures (topdown-style rendering path),
    # while overlays use a baked vertex-color copy for stable tinting.
    # Keep the full (pre-roof-removal) mesh for raycasting and IoU tinting so
    # that triangle indices are consistent across raycasting and overlay building.
    full_mesh_display, full_mesh_tint = load_textured_mesh(scene_dir, dataset)

    # Use the same mesh for raycasting — avoids index mismatches with a
    # separately-loaded mesh and ensures IoU overlays get proper vertex colors.
    rc = o3d.t.geometry.RaycastingScene()
    mesh_id = rc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(full_mesh_display))

    verts = np.asarray(full_mesh_display.vertices)
    tris = np.asarray(full_mesh_display.triangles)
    tri_pts = verts[tris]
    tri_centroids = tri_pts.mean(axis=1)

    # Display mesh: optionally remove roof for better top-down visibility
    display_mesh = _remove_roof(full_mesh_display, z_percentile=roof_z_percentile) if remove_roof else full_mesh_display
    overlay_source_mesh = full_mesh_tint

    # Compute visible triangle sets
    gt_vis = _visible_triangles_from_view(
        gt_position, gt_direction, hfov_rad, vfov_rad,
        rc, int(mesh_id), tri_centroids,
    )
    pred_vis = _visible_triangles_from_view(
        pred_position, pred_direction, hfov_rad, vfov_rad,
        rc, int(mesh_id), tri_centroids,
    )

    gt_only = gt_vis - pred_vis
    pred_only = pred_vis - gt_vis
    both = gt_vis & pred_vis

    union = len(gt_vis | pred_vis)
    iou = len(both) / union if union > 0 else 0.0
    logger.info(f"IoU: {iou:.3f} (GT: {len(gt_vis)} tris, Pred: {len(pred_vis)} tris, "
                f"intersection: {len(both)}, union: {union})")

    dist_err = float(np.linalg.norm(pred_position - gt_position))
    ang_err_deg: Optional[float] = None
    if gt_direction is not None and pred_direction is not None:
        dot = float(np.clip(np.dot(
            gt_direction / np.linalg.norm(gt_direction),
            pred_direction / np.linalg.norm(pred_direction),
        ), -1, 1))
        ang_err_deg = math.degrees(math.acos(dot))

    # For 3RScan, use legacy viewer like topdown_3rscan.py so UV textures are
    # rendered correctly. SceneWidget path is retained for ScanNet.
    if dataset == "3rscan":
        geoms: List[o3d.geometry.Geometry] = []
        if show_base_textured:
            geoms.append(display_mesh)

        tint_gt_only = np.array([1.0, 0.1, 0.1])
        tint_pred_only = np.array([1.0, 0.9, 0.0])
        tint_both = np.array([1.0, 0.5, 0.0])
        overlays = [
            ("iou_gt_only", gt_only, tint_gt_only, tint_strength),
            ("iou_pred_only", pred_only, tint_pred_only, tint_strength),
            ("iou_both", both, tint_both, max(tint_strength, 0.90)),
        ]
        for _, tri_set, tint_colour, local_strength in overlays:
            overlay_mesh = _build_tinted_overlay(
                overlay_source_mesh, tri_set, tint_colour, tint_strength=local_strength
            )
            if overlay_mesh is not None:
                geoms.append(overlay_mesh)

        gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=gt_sphere_radius)
        gt_sphere.translate(gt_position)
        gt_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        gt_sphere.compute_vertex_normals()
        geoms.append(gt_sphere)

        pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=pred_sphere_radius)
        pred_sphere.translate(pred_position)
        pred_sphere.paint_uniform_color([1.0, 0.85, 0.0])
        pred_sphere.compute_vertex_normals()
        geoms.append(pred_sphere)

        frustum_gt = create_camera_frustum(
            gt_position, gt_direction, colour=(1.0, 0.0, 0.0),
            h_fov=hfov_rad, v_fov=vfov_rad, scale=frustum_scale
        )
        if frustum_gt is not None:
            geoms.append(frustum_gt)

        frustum_pred = create_camera_frustum(
            pred_position, pred_direction, colour=(1.0, 0.85, 0.0),
            h_fov=hfov_rad, v_fov=vfov_rad, scale=frustum_scale
        )
        if frustum_pred is not None:
            geoms.append(frustum_pred)

        err_line = o3d.geometry.LineSet()
        err_line.points = o3d.utility.Vector3dVector(np.array([gt_position, pred_position]))
        err_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
        err_line.paint_uniform_color([1.0, 1.0, 1.0])
        geoms.append(err_line)

        title = f"{scene_id}"
        if frame_id:
            title += f" / frame {frame_id}"
        title += f" — IoU: {iou:.3f}"
        o3d.visualization.draw_geometries(
            geoms,
            window_name=title,
            width=1440,
            height=900,
            mesh_show_back_face=True,
        )
        return

    # --- Build Open3D visualizer using gui.Window for 2D legend overlay ---
    from open3d.visualization import gui, rendering

    app = gui.Application.instance
    app.initialize()

    title = f"{scene_id}"
    if frame_id:
        title += f" / frame {frame_id}"
    title += f" — IoU: {iou:.3f}"

    window = gui.Application.instance.create_window(title, 1440, 900)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background([0.9, 0.9, 0.9, 1.0])

    # --- Legend and description panels (2D overlay, toggleable via show_legend) ---
    em = window.theme.font_size
    legend_panel = None
    desc_panel = None

    if show_legend:
        legend_panel = gui.Vert(int(0.3 * em), gui.Margins(int(0.6 * em),
                                                             int(0.6 * em),
                                                             int(0.6 * em),
                                                             int(0.6 * em)))
        legend_panel.background_color = gui.Color(0.0, 0.0, 0.0, 0.7)

        def _legend_label(text: str, color: gui.Color) -> gui.Label:
            lbl = gui.Label(text)
            lbl.text_color = color
            return lbl

        legend_panel.add_child(_legend_label(
            f"Dataset: {dataset}  |  Scene: {scene_id}  |  Frame: {frame_id or 'n/a'}",
            gui.Color(0.85, 0.85, 0.85),
        ))
        legend_panel.add_child(_legend_label(
            f"IoU: {iou:.3f}  |  dist err: {dist_err:.2f}m"
            + (f"  |  ang err: {ang_err_deg:.1f}" if ang_err_deg is not None else ""),
            gui.Color(1.0, 1.0, 1.0),
        ))
        legend_panel.add_child(_legend_label("[o] GT Camera (red frustum)",
                                              gui.Color(1.0, 0.3, 0.3)))
        legend_panel.add_child(_legend_label("[o] Predicted Camera (yellow frustum)",
                                              gui.Color(1.0, 0.9, 0.2)))
        legend_panel.add_child(_legend_label("[=] GT only visible (red tint)",
                                              gui.Color(1.0, 0.3, 0.3)))
        legend_panel.add_child(_legend_label("[=] Pred only visible (yellow tint)",
                                              gui.Color(1.0, 0.9, 0.2)))
        legend_panel.add_child(_legend_label("[=] Overlap (orange tint)",
                                              gui.Color(1.0, 0.6, 0.15)))

        # --- Description panel (top-right) ---
        if description:
            desc_panel = gui.Vert(int(0.3 * em), gui.Margins(int(0.6 * em),
                                                               int(0.6 * em),
                                                               int(0.6 * em),
                                                               int(0.6 * em)))
            desc_panel.background_color = gui.Color(0.0, 0.0, 0.0, 0.7)

            desc_title = gui.Label("Input View Description:")
            desc_title.text_color = gui.Color(1.0, 0.85, 0.3)
            desc_panel.add_child(desc_title)

            words = description.split()
            lines: List[str] = []
            current = ""
            for w in words:
                if current and len(current) + 1 + len(w) > desc_wrap_width:
                    lines.append(current)
                    current = w
                else:
                    current = current + " " + w if current else w
            if current:
                lines.append(current)

            for line in lines:
                lbl = gui.Label(line)
                lbl.text_color = gui.Color(0.9, 0.9, 0.9)
                desc_panel.add_child(lbl)

    window.add_child(scene_widget)
    if legend_panel is not None:
        window.add_child(legend_panel)
    if desc_panel is not None:
        window.add_child(desc_panel)

    def _on_layout(ctx):
        r = window.content_rect
        scene_widget.frame = r
        if legend_panel is not None:
            pref = legend_panel.calc_preferred_size(ctx, gui.Widget.Constraints())
            legend_panel.frame = gui.Rect(
                int(r.x + 10),
                int(r.y + 10),
                int(pref.width),
                int(pref.height),
            )
        if desc_panel is not None:
            dp = desc_panel.calc_preferred_size(ctx, gui.Widget.Constraints())
            desc_panel.frame = gui.Rect(
                int(r.x + r.width - dp.width - 10),
                int(r.y + 10),
                int(dp.width),
                int(dp.height),
            )

    window.set_on_layout(_on_layout)

    # --- Add 3D geometry to the scene ---
    mat_textured = rendering.MaterialRecord()
    mat_textured.shader = "defaultLit"
    mat_textured.base_color = [1.0, 1.0, 1.0, 1.0]

    if show_base_textured:
        scene_widget.scene.add_geometry("mesh_textured", display_mesh, mat_textured)

    # IoU tinted overlays — red=GT only, yellow=pred only, orange=overlap
    # Use full_mesh (pre-roof-removal, with baked vertex colors) so triangle
    # indices from raycasting match and overlays get proper texture colors.
    tint_gt_only = np.array([1.0, 0.1, 0.1])
    tint_pred_only = np.array([1.0, 0.9, 0.0])
    tint_both = np.array([1.0, 0.5, 0.0])

    base_for_tint = overlay_source_mesh

    overlays = [
        ("iou_gt_only", gt_only, tint_gt_only, tint_strength),
        ("iou_pred_only", pred_only, tint_pred_only, tint_strength),
        # Make overlap more visible than the single-view sets.
        ("iou_both", both, tint_both, max(tint_strength, 0.90)),
    ]

    for name, tri_set, tint_colour, local_strength in overlays:
        overlay_mesh = _build_tinted_overlay(base_for_tint, tri_set, tint_colour,
                                             tint_strength=local_strength)
        if overlay_mesh is None:
            continue
        mat = rendering.MaterialRecord()
        mat.shader = "defaultUnlit" if name == "iou_both" else "defaultLit"
        scene_widget.scene.add_geometry(name, overlay_mesh, mat)

    # GT camera sphere — RED
    gt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=gt_sphere_radius)
    gt_sphere.translate(gt_position)
    gt_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    gt_sphere.compute_vertex_normals()
    mat_sphere = rendering.MaterialRecord()
    mat_sphere.shader = "defaultLit"
    scene_widget.scene.add_geometry("gt_camera", gt_sphere, mat_sphere)

    # Predicted camera sphere — YELLOW
    pred_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=pred_sphere_radius)
    pred_sphere.translate(pred_position)
    pred_sphere.paint_uniform_color([1.0, 0.85, 0.0])
    pred_sphere.compute_vertex_normals()
    scene_widget.scene.add_geometry("pred_camera", pred_sphere, mat_sphere)

    # GT frustum — RED
    frustum_gt = create_camera_frustum(gt_position, gt_direction,
                                       colour=(1.0, 0.0, 0.0),
                                       h_fov=hfov_rad, v_fov=vfov_rad,
                                       scale=frustum_scale)
    if frustum_gt is not None:
        mat_frustum_gt = rendering.MaterialRecord()
        mat_frustum_gt.shader = "unlitLine"
        mat_frustum_gt.line_width = 3.5
        mat_frustum_gt.base_color = [1.0, 0.0, 0.0, 1.0]
        scene_widget.scene.add_geometry("frustum_gt", frustum_gt, mat_frustum_gt)

    # Predicted frustum — YELLOW
    frustum_pred = create_camera_frustum(pred_position, pred_direction,
                                         colour=(1.0, 0.85, 0.0),
                                         h_fov=hfov_rad, v_fov=vfov_rad,
                                         scale=frustum_scale)
    if frustum_pred is not None:
        mat_frustum_pred = rendering.MaterialRecord()
        mat_frustum_pred.shader = "unlitLine"
        mat_frustum_pred.line_width = 3.5
        mat_frustum_pred.base_color = [1.0, 0.85, 0.0, 1.0]
        scene_widget.scene.add_geometry("frustum_pred", frustum_pred, mat_frustum_pred)

    # Distance error line between GT and predicted
    err_line = o3d.geometry.LineSet()
    err_line.points = o3d.utility.Vector3dVector(
        np.array([gt_position, pred_position])
    )
    err_line.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    err_line.paint_uniform_color([1.0, 1.0, 1.0])
    mat_err = rendering.MaterialRecord()
    mat_err.shader = "unlitLine"
    mat_err.line_width = 2.0
    mat_err.base_color = [1.0, 1.0, 1.0, 1.0]
    scene_widget.scene.add_geometry("error_line", err_line, mat_err)

    # Fit camera to scene bounds
    bounds = scene_widget.scene.bounding_box
    scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    app.run()


# --------------------------------------------------------------------------- #
# CLI: load from candidates JSON                                              #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize IoU with textured mesh from mk5 candidates JSON."
    )
    parser.add_argument("--candidates", required=True, type=Path,
                        help="Path to candidates JSON from visualize_eval_loc_mk5.py")
    parser.add_argument("--root", required=True, type=Path,
                        help="Root directory containing <scene_id>/ meshes.")
    parser.add_argument("--dataset", required=True, choices=["3rscan", "scannet"])
    parser.add_argument("--scene_id", required=True,
                        help="Scene ID to visualize.")
    parser.add_argument("--frame_id", default=None,
                        help="Specific frame ID (if multiple frames per scene).")
    parser.add_argument("--h_fov_deg", type=float, default=39.31)
    parser.add_argument("--v_fov_deg", type=float, default=64.76)
    parser.add_argument("--frustum_scale", type=float, default=0.8)
    parser.add_argument("--tint_strength", type=float, default=0.45,
                        help="How strongly to tint IoU overlay regions (0=invisible, 1=solid).")
    parser.add_argument("--no_base_mesh", action="store_true",
                        help="Hide the base textured mesh (show only IoU overlays).")
    parser.add_argument("--no_remove_roof", action="store_true",
                        help="Keep the roof/ceiling (by default it is removed).")
    parser.add_argument("--roof_z_percentile", type=float, default=90.0,
                        help="Z-percentile above which triangles are removed as roof (default 90).")
    parser.add_argument("--desc_wrap_width", type=int, default=50,
                        help="Character width for word-wrapping the description panel (default 50).")
    parser.add_argument("--legend", action="store_true", default=False,
                        help="Show legend and description panels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    cand_data = json.loads(args.candidates.read_text())
    scenes = cand_data.get("scenes", [])

    # Find the requested scene
    entry = None
    for s in scenes:
        if s["scene_id"] != args.scene_id:
            continue
        if args.frame_id is not None and str(s.get("frame_id")) != args.frame_id:
            continue
        entry = s
        break

    if entry is None:
        available = [s["scene_id"] for s in scenes]
        logger.error(f"Scene '{args.scene_id}' (frame={args.frame_id}) not found in candidates. "
                     f"Available: {available[:20]}...")
        return

    gt = entry["gt_pose"]
    pred = entry["predicted_pose"]
    gt_pos = np.asarray(gt["position"], dtype=np.float64)
    gt_dir = np.asarray(gt["direction"], dtype=np.float64) if gt.get("direction") else None
    pred_pos = np.asarray(pred["position"], dtype=np.float64)
    pred_dir = np.asarray(pred["direction"], dtype=np.float64) if pred.get("direction") else None

    scene_dir = args.root / args.scene_id
    if not scene_dir.exists():
        logger.error(f"Scene directory not found: {scene_dir}")
        return

    logger.info(f"Scene: {args.scene_id} | Frame: {entry.get('frame_id', 'n/a')}")
    logger.info(f"GT position:   {gt_pos.tolist()}")
    logger.info(f"Pred position: {pred_pos.tolist()}")
    dist = float(np.linalg.norm(pred_pos - gt_pos))
    logger.info(f"Distance error: {dist:.3f} m")
    if gt_dir is not None and pred_dir is not None:
        dot = float(np.clip(np.dot(gt_dir / np.linalg.norm(gt_dir),
                                    pred_dir / np.linalg.norm(pred_dir)), -1, 1))
        logger.info(f"Angular error:  {math.degrees(math.acos(dot)):.2f} deg")

    visualize_scene_iou(
        scene_dir=scene_dir,
        dataset=args.dataset,
        gt_position=gt_pos,
        gt_direction=gt_dir,
        pred_position=pred_pos,
        pred_direction=pred_dir,
        scene_id=args.scene_id,
        frame_id=str(entry.get("frame_id", "")),
        h_fov_deg=args.h_fov_deg,
        v_fov_deg=args.v_fov_deg,
        frustum_scale=args.frustum_scale,
        tint_strength=args.tint_strength,
        show_base_textured=not args.no_base_mesh,
        remove_roof=not args.no_remove_roof,
        roof_z_percentile=args.roof_z_percentile,
        description=entry.get("description", "") or "",
        desc_wrap_width=args.desc_wrap_width,
        show_legend=args.legend,
    )


if __name__ == "__main__":
    main()
