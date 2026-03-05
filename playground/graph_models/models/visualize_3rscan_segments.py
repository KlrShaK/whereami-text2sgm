#!/usr/bin/env python3
"""
Utility helpers to visualise 3RScan meshes with instance segmentation overlays.

The module exposes reusable functions that can be imported by other scripts
(e.g., visualize_eval_loc.py) while still supporting the original CLI viewer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import plyfile
from sklearn.neighbors import NearestNeighbors

try:  # Optional dependency; only needed for the CLI entry point.
    from src.utils.config_loader import load_config  # type: ignore
except ImportError:  # pragma: no cover
    load_config = None  # type: ignore[assignment]

__all__ = [
    "build_segmented_mesh",
    "create_segment_visualizer",
]


def _load_segmentation(scene_path: Path, base_vertices: np.ndarray,
                       dataset: str = "3rscan") -> Tuple[np.ndarray, Dict[int, int], Dict[int, str]]:
    """
    Assign instance IDs and semantic labels to each vertex of the textured mesh.

    Supports both 3RScan (via annotated PLY + semseg JSON) and ScanNet (via
    segment-indices JSON + aggregation JSON).

    Args:
        scene_path (Path): Scene directory.
        base_vertices (np.ndarray): (N, 3) array of the mesh vertices.
        dataset (str): ``"3rscan"`` or ``"scannet"``.

    Returns:
        tuple[np.ndarray, dict[int, int], dict[int, str]]:
            - vert_obj: per-vertex instance IDs aligned with `base_vertices`.
            - seg_to_obj: segment-to-object mapping.
            - obj_to_label: objectId → human-readable label.

    Raises:
        FileNotFoundError: If required annotation files are missing.
    """
    if dataset == "scannet":
        return _load_segmentation_scannet(scene_path, base_vertices)

    # --- 3RScan path (original) ---
    semseg_json = scene_path / "semseg.v2.json"
    ply_path = scene_path / "labels.instances.annotated.v2.ply"
    if not semseg_json.exists() or not ply_path.exists():
        raise FileNotFoundError(f"Missing semantic annotations or annotated point cloud next to {scene_path}")

    groups = json.loads(semseg_json.read_text())["segGroups"]
    obj_to_label: Dict[int, str] = {int(g["objectId"]): g.get("label", "").strip() for g in groups}

    ply = plyfile.PlyData.read(ply_path)
    pts = np.vstack([ply["vertex"][axis] for axis in ("x", "y", "z")]).T.astype(np.float32)
    obj_ids = np.asarray(ply["vertex"]["objectId"], dtype=np.int32)

    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(pts)
    _, idx = nn.kneighbors(base_vertices.astype(np.float32), return_distance=True)
    vert_obj = obj_ids[idx[:, 0]]

    seg_to_obj = {int(oid): int(oid) for oid in np.unique(vert_obj) if oid >= 0}
    return vert_obj.astype(np.int32), seg_to_obj, obj_to_label


def _load_segmentation_scannet(scene_path: Path, base_vertices: np.ndarray
                               ) -> Tuple[np.ndarray, Dict[int, int], Dict[int, str]]:
    """ScanNet segmentation: segment-indices + aggregation JSONs."""
    sid = scene_path.name
    segs_path = scene_path / f"{sid}_vh_clean_2.0.010000.segs.json"
    agg_path = scene_path / f"{sid}.aggregation.json"

    if not segs_path.exists():
        raise FileNotFoundError(f"Missing ScanNet segment indices: {segs_path}")
    if not agg_path.exists():
        raise FileNotFoundError(f"Missing ScanNet aggregation metadata: {agg_path}")

    seg_indices_raw = json.loads(segs_path.read_text()).get("segIndices", [])
    seg_indices = np.asarray(seg_indices_raw, dtype=np.int64)
    if len(seg_indices) != len(base_vertices):
        raise ValueError(
            f"ScanNet segmentation length mismatch for '{sid}': "
            f"segIndices={len(seg_indices)} vs vertices={len(base_vertices)}"
        )

    agg_data = json.loads(agg_path.read_text())
    seg_groups = agg_data.get("segGroups", [])

    seg_to_obj: Dict[int, int] = {}
    obj_to_label: Dict[int, str] = {}
    for group in seg_groups:
        if not isinstance(group, dict):
            continue
        try:
            object_id = int(group.get("objectId", group.get("id", 0)))
        except (TypeError, ValueError):
            continue
        obj_to_label[object_id] = group.get("label", "").strip()
        for seg in group.get("segments", []):
            try:
                seg_to_obj[int(seg)] = object_id
            except (TypeError, ValueError):
                continue

    vert_obj = np.array(
        [seg_to_obj.get(int(s), 0) for s in seg_indices], dtype=np.int32
    )
    return vert_obj, seg_to_obj, obj_to_label


def _find_scannet_mesh_path(scene_path: Path) -> Path:
    """Locate the ScanNet clean mesh PLY inside *scene_path*."""
    sid = scene_path.name
    for name in (f"{sid}_vh_clean_2.ply", f"{sid}_vh_clean_2.labels.ply"):
        p = scene_path / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No ScanNet mesh found in {scene_path}")


def build_segmented_mesh(scene_path: Path,
                         seed: int = 7,
                         only_ids: Optional[Sequence[int]] = None,
                         dataset: str = "3rscan") -> Tuple[o3d.geometry.TriangleMesh, List[Dict[str, object]]]:
    """
    Construct an Open3D mesh with per-vertex colors encoding instance IDs.

    The function duplicates vertices per triangle so that each face can be
    rendered with an unambiguous color. It also computes per-object statistics
    (centroid, bbox, palette) used by the viewer for overlays and labels.

    Args:
        scene_path (Path): Scene directory (3RScan or ScanNet).
        seed (int): Random seed for color palette.
        only_ids: Optional subset of object IDs to include in stats.
        dataset (str): ``"3rscan"`` or ``"scannet"``.

    Returns:
        tuple[o3d.geometry.TriangleMesh, list[dict[str, object]]]:
            - Colored mesh ready to be displayed.
            - List of per-object metadata dictionaries.
    """
    if dataset == "scannet":
        mesh_path = _find_scannet_mesh_path(scene_path)
    else:
        mesh_path = scene_path / "mesh.refined.v2.obj"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=True)
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    expanded_verts = verts[faces.reshape(-1)]
    expanded_faces = np.arange(len(expanded_verts), dtype=np.int32).reshape(-1, 3)

    vert_seg_raw, seg_to_obj, obj_to_label = _load_segmentation(scene_path, verts, dataset=dataset)
    vert_obj = vert_seg_raw[faces.reshape(-1)]

    unique_obj_ids = sorted({oid for oid in vert_obj if oid >= 0})
    rng = np.random.default_rng(seed)
    palette = {oid: rng.uniform(0.15, 0.95, size=3) for oid in unique_obj_ids}

    colors = np.zeros((expanded_verts.shape[0], 3), dtype=np.float64)
    for oid in unique_obj_ids:
        colors[vert_obj == oid] = palette[oid]
    colors[vert_obj < 0] = np.array([0.6, 0.6, 0.6])

    mesh_vis = o3d.geometry.TriangleMesh()
    mesh_vis.vertices = o3d.utility.Vector3dVector(expanded_verts)
    mesh_vis.triangles = o3d.utility.Vector3iVector(expanded_faces)
    mesh_vis.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh_vis.compute_vertex_normals()

    obj_stats: List[Dict[str, object]] = []
    for oid in unique_obj_ids:
        vert_idx = np.nonzero(vert_obj == oid)[0]
        if vert_idx.size == 0:
            continue
        if only_ids and oid not in only_ids:
            continue
        obj_vertices = expanded_verts[vert_idx]
        centroid = obj_vertices.mean(axis=0)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            obj_vertices.min(axis=0), obj_vertices.max(axis=0)
        )
        bbox.color = palette[oid]
        obj_stats.append(
            {
                "object_id": oid,
                "label": obj_to_label.get(oid, ""),
                "centroid": centroid,
                "bbox": bbox,
                "color": palette[oid],
                "vertex_indices": vert_idx,
            }
        )
    return mesh_vis, obj_stats


def create_segment_visualizer(mesh: o3d.geometry.TriangleMesh,
                              obj_stats: Sequence[Dict[str, object]],
                              *,
                              highlight_ids: Optional[Iterable[int]] = None,
                              show_bboxes: bool = True,
                              window_name: str = "3RScan Segmentation") -> o3d.visualization.O3DVisualizer:
    """
    Create an Open3D O3DVisualizer instance populated with coloured segments,
    per-object bounding boxes, and 3D labels.
    """
    from open3d.visualization import gui, rendering

    highlight_ids = set(highlight_ids or [])

    material = rendering.MaterialRecord()
    material.shader = "defaultLit"

    line_material = rendering.MaterialRecord()
    line_material.shader = "unlitLine"
    line_material.line_width = 1.0

    vis = o3d.visualization.O3DVisualizer(window_name, 1280, 720)
    vis.show_settings = False
    vis.add_geometry("mesh", mesh, material)

    for stats in obj_stats:
        oid = int(stats["object_id"])
        label = stats.get("label") or f"id_{oid}"
        colour = stats.get("color", (0.8, 0.8, 0.8))
        centroid = np.asarray(stats["centroid"])

        if show_bboxes and "bbox" in stats:
            bbox: o3d.geometry.AxisAlignedBoundingBox = stats["bbox"]
            vis.add_geometry(f"bbox_{oid}", bbox, line_material)

        vis.add_3d_label(centroid, f"{oid}: {label}")

        if highlight_ids and oid in highlight_ids:
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
            marker.translate(centroid)
            marker.paint_uniform_color(np.asarray(colour))
            if not marker.has_vertex_normals():
                marker.compute_vertex_normals()
            vis.add_geometry(f"marker_{oid}", marker, material)

    vis.reset_camera_to_default()
    gui.Application.instance.add_window(vis)
    return vis


def parse_args():
    """Parse command-line arguments for the visualization script."""
    parser = argparse.ArgumentParser(description="Visualize 3RScan instance labels in 3D.")
    parser.add_argument("--scene", required=True, help="3RScan scene folder name.")
    parser.add_argument("--root", type=Path, help="Path to the 3RScan dataset root.")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Project config path (fallback if --root missing).")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for colors.")
    parser.add_argument(
        "--only-ids", type=int, nargs="*", help="Optional list of object IDs to annotate."
    )
    parser.add_argument("--no-bboxes", action="store_true", help="Disable bounding boxes.")
    return parser.parse_args()


def main():
    """Program entry: load configuration, build mesh, and start the viewer."""
    args = parse_args()
    scene_root = args.root
    if scene_root is None:
        if load_config is None:
            raise RuntimeError("Either --root must be provided or load_config must be available.")
        cfg = load_config(args.config)
        scene_root = Path(cfg["paths"]["3rscan_dataset_path"]).expanduser()

    dataset_root = scene_root
    scene_path = dataset_root / args.scene
    if not scene_path.exists():
        raise FileNotFoundError(scene_path)

    mesh_vis, obj_stats = build_segmented_mesh(scene_path, seed=args.seed, only_ids=args.only_ids)

    from open3d.visualization import gui

    gui.Application.instance.initialize()
    vis = create_segment_visualizer(
        mesh_vis, obj_stats,
        highlight_ids=args.only_ids,
        show_bboxes=not args.no_bboxes,
    )
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
