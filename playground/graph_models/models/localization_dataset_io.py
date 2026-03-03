from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

DatasetName = Literal["3rscan", "scannet"]


@dataclass(frozen=True)
class FrameSelectionLike:
    frame: dict
    path: Path


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _try_load_json(path: Path) -> Optional[object]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _build_obj2faces(tri2obj: np.ndarray) -> Dict[int, np.ndarray]:
    obj2faces: Dict[int, List[int]] = {}
    for face_id, oid in enumerate(tri2obj):
        oid_i = int(oid)
        if oid_i <= 0:
            continue
        obj2faces.setdefault(oid_i, []).append(face_id)
    return {k: np.asarray(v, dtype=np.int32) for k, v in obj2faces.items()}


def _majority_object_id(vertex_object_ids: np.ndarray) -> int:
    valid = vertex_object_ids[vertex_object_ids >= 0]
    if valid.size == 0:
        return 0
    counts = np.bincount(valid)
    return int(np.argmax(counts))


def _load_3rscan_scene_geometry(scene_dir: Path) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, Dict[int, np.ndarray]]:
    import open3d as o3d

    ply_path = scene_dir / "labels.instances.annotated.v2.ply"
    if not ply_path.exists():
        raise FileNotFoundError(
            f"Missing 3RScan scene mesh for '{scene_dir.name}': {ply_path}"
        )

    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertex_colors = np.asarray(mesh.vertex_colors)
    if vertex_colors.size == 0:
        raise ValueError(
            f"3RScan mesh has no vertex colors for '{scene_dir.name}': {ply_path}"
        )

    objects_path = scene_dir.parent / "objects.json"
    objects_payload = _load_json(objects_path)
    scans = objects_payload.get("scans", []) if isinstance(objects_payload, dict) else []
    scans_by_id = {str(item.get("scan", "")): item for item in scans if isinstance(item, dict)}
    if scene_dir.name not in scans_by_id:
        raise KeyError(
            f"Scene '{scene_dir.name}' not found in 3RScan objects metadata: {objects_path}"
        )

    scan_meta = scans_by_id[scene_dir.name]
    color_to_oid: Dict[int, int] = {}
    for obj in scan_meta.get("objects", []):
        if not isinstance(obj, dict):
            continue
        color = str(obj.get("ply_color", "")).strip().lstrip("#")
        if not color:
            continue
        try:
            color_int = int(color, 16)
            oid = int(obj.get("id", 0))
        except (TypeError, ValueError):
            continue
        color_to_oid[color_int] = oid

    rgb_u8 = (vertex_colors * 255.0 + 0.5).astype(np.uint32)
    vertex_hex = (rgb_u8[:, 0] << 16) | (rgb_u8[:, 1] << 8) | rgb_u8[:, 2]
    vertex_oids = np.array([color_to_oid.get(int(h), 0) for h in vertex_hex], dtype=np.int32)

    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    tri2obj = np.array(
        [_majority_object_id(vertex_oids[tri]) for tri in triangles],
        dtype=np.int32,
    )
    obj2faces = _build_obj2faces(tri2obj)
    return mesh, tri2obj, obj2faces


def _find_scannet_mesh_path(scene_dir: Path) -> Path:
    sid = scene_dir.name
    candidates = [
        scene_dir / f"{sid}_vh_clean_2.ply",
        scene_dir / f"{sid}_vh_clean_2.labels.ply",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing ScanNet mesh for '{sid}'. Checked: {', '.join(str(p) for p in candidates)}"
    )


def _load_scannet_scene_geometry(scene_dir: Path) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, Dict[int, np.ndarray]]:
    import open3d as o3d

    sid = scene_dir.name
    mesh_path = _find_scannet_mesh_path(scene_dir)
    segs_path = scene_dir / f"{sid}_vh_clean_2.0.010000.segs.json"
    agg_path = scene_dir / f"{sid}.aggregation.json"

    if not segs_path.exists():
        raise FileNotFoundError(
            f"Missing ScanNet segment indices for '{sid}': {segs_path}"
        )
    if not agg_path.exists():
        raise FileNotFoundError(
            f"Missing ScanNet aggregation metadata for '{sid}': {agg_path}"
        )

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)

    segs_payload = _load_json(segs_path)
    seg_indices_raw = segs_payload.get("segIndices", []) if isinstance(segs_payload, dict) else []
    if not isinstance(seg_indices_raw, list):
        raise ValueError(f"Expected list segIndices in {segs_path}")
    if len(seg_indices_raw) != len(vertices):
        raise ValueError(
            "ScanNet vertex segmentation length mismatch for "
            f"'{sid}': segIndices={len(seg_indices_raw)} vs vertices={len(vertices)}"
        )
    seg_indices = np.asarray(seg_indices_raw, dtype=np.int64)

    agg_payload = _load_json(agg_path)
    seg_groups = agg_payload.get("segGroups", []) if isinstance(agg_payload, dict) else []
    seg_to_oid: Dict[int, int] = {}
    for group in seg_groups:
        if not isinstance(group, dict):
            continue
        try:
            object_id = int(group.get("objectId", group.get("id", 0)))
        except (TypeError, ValueError):
            continue
        segments = group.get("segments", [])
        if not isinstance(segments, list):
            continue
        for seg in segments:
            try:
                seg_to_oid[int(seg)] = object_id
            except (TypeError, ValueError):
                continue

    vertex_oids = np.array(
        [seg_to_oid.get(int(seg_id), 0) for seg_id in seg_indices],
        dtype=np.int32,
    )
    tri2obj = np.array(
        [_majority_object_id(vertex_oids[tri]) for tri in triangles],
        dtype=np.int32,
    )
    obj2faces = _build_obj2faces(tri2obj)
    return mesh, tri2obj, obj2faces


def load_scene_geometry(
    scene_dir: Path, dataset: DatasetName
) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, Dict[int, np.ndarray]]:
    if dataset == "3rscan":
        return _load_3rscan_scene_geometry(scene_dir)
    if dataset == "scannet":
        return _load_scannet_scene_geometry(scene_dir)
    raise ValueError(f"Unsupported dataset '{dataset}' for scene '{scene_dir.name}'")


def _load_3rscan_object_labels(scene_dir: Path) -> Dict[int, str]:
    semseg_path = scene_dir / "semseg.v2.json"
    if not semseg_path.exists():
        return {}
    payload = _load_json(semseg_path)
    seg_groups = payload.get("segGroups", []) if isinstance(payload, dict) else []
    labels: Dict[int, str] = {}
    for group in seg_groups:
        if not isinstance(group, dict):
            continue
        try:
            oid = int(group.get("objectId", group.get("id", 0)))
        except (TypeError, ValueError):
            continue
        labels[oid] = str(group.get("label", "")).strip()
    return labels


def _load_scannet_object_labels(scene_dir: Path) -> Dict[int, str]:
    sid = scene_dir.name
    agg_path = scene_dir / f"{sid}.aggregation.json"
    if not agg_path.exists():
        return {}
    payload = _load_json(agg_path)
    seg_groups = payload.get("segGroups", []) if isinstance(payload, dict) else []
    labels: Dict[int, str] = {}
    for group in seg_groups:
        if not isinstance(group, dict):
            continue
        try:
            oid = int(group.get("objectId", group.get("id", 0)))
        except (TypeError, ValueError):
            continue
        labels[oid] = str(group.get("label", "")).strip()
    return labels


def load_object_labels(scene_dir: Path, dataset: DatasetName) -> Dict[int, str]:
    if dataset == "3rscan":
        return _load_3rscan_object_labels(scene_dir)
    if dataset == "scannet":
        return _load_scannet_object_labels(scene_dir)
    raise ValueError(f"Unsupported dataset '{dataset}' for scene '{scene_dir.name}'")


def _tokenize_label(label: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(label).lower())


def load_floor_object_ids(scene_dir: Path, dataset: DatasetName) -> set[int]:
    labels = load_object_labels(scene_dir, dataset)
    floor_ids: set[int] = set()
    for oid, label in labels.items():
        if "floor" in _tokenize_label(label):
            floor_ids.add(int(oid))
    return floor_ids


def _expand_json_payload(data: object, path: Path) -> List[FrameSelectionLike]:
    if isinstance(data, dict):
        return [FrameSelectionLike(frame=data, path=path)]
    if isinstance(data, list):
        records: List[FrameSelectionLike] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            virtual_name = path.with_name(f"{path.stem}_{idx:03d}{path.suffix}")
            records.append(FrameSelectionLike(frame=item, path=virtual_name))
        return records
    return []


def _looks_like_structured_frame(data: dict) -> bool:
    has_identity = "image_index" in data or "scene_pose" in data
    has_payload = "description" in data or "visible_objects" in data or "scene_pose" in data
    return has_identity and has_payload


def _dedupe_frames(frames: Sequence[FrameSelectionLike]) -> List[FrameSelectionLike]:
    deduped: List[FrameSelectionLike] = []
    seen_image_indices: set[str] = set()
    for fs in frames:
        image_index = str(fs.frame.get("image_index", "")).strip()
        if image_index:
            if image_index in seen_image_indices:
                continue
            seen_image_indices.add(image_index)
        deduped.append(fs)
    return deduped


def load_structured_frame_jsons(desc_dir: Path) -> List[FrameSelectionLike]:
    if not desc_dir.exists():
        return []

    all_json = sorted(desc_dir.glob("*.json"))
    non_parsed_json = [p for p in all_json if not p.stem.endswith("_parsed")]
    if not non_parsed_json:
        return []

    explicit_frame_records: List[FrameSelectionLike] = []
    for path in non_parsed_json:
        if path.name == "all_descriptions.json":
            continue
        payload = _try_load_json(path)
        if not isinstance(payload, dict):
            continue
        if _looks_like_structured_frame(payload):
            explicit_frame_records.append(FrameSelectionLike(frame=payload, path=path))

    if explicit_frame_records:
        return _dedupe_frames(explicit_frame_records)

    candidate_paths: List[Path]
    all_desc_path = desc_dir / "all_descriptions.json"
    if all_desc_path.exists():
        candidate_paths = [all_desc_path]
    else:
        candidate_paths = non_parsed_json

    records: List[FrameSelectionLike] = []
    for path in candidate_paths:
        payload = _try_load_json(path)
        if payload is None:
            continue
        records.extend(_expand_json_payload(payload, path))
    return _dedupe_frames(records)


def load_parsed_frame_jsons(desc_dir: Path) -> List[FrameSelectionLike]:
    if not desc_dir.exists():
        return []

    records: List[FrameSelectionLike] = []
    for path in sorted(desc_dir.glob("*_parsed.json")):
        payload = _try_load_json(path)
        if isinstance(payload, dict) and "parsed_graph" in payload:
            records.append(FrameSelectionLike(frame=payload, path=path))
    return records


def resolve_raw_frame_path_from_parsed(parsed_path: Path) -> Path:
    if parsed_path.suffix.lower() != ".json":
        raise ValueError(f"Expected parsed JSON path, got: {parsed_path}")

    parsed_suffix = "_parsed"
    stem = parsed_path.stem
    if not stem.endswith(parsed_suffix):
        raise ValueError(f"Path does not end with '{parsed_suffix}': {parsed_path}")

    raw_stem = stem[: -len(parsed_suffix)]
    if not raw_stem:
        raise ValueError(f"Invalid parsed frame path: {parsed_path}")
    return parsed_path.with_name(f"{raw_stem}.json")
