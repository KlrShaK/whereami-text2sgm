from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))

from localization_dataset_io import (  # noqa: E402
    load_floor_object_ids,
    load_object_labels,
    load_parsed_frame_jsons,
    load_structured_frame_jsons,
    resolve_raw_frame_path_from_parsed,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_load_structured_frame_jsons_prefers_explicit_frames(tmp_path: Path) -> None:
    desc_dir = tmp_path / "output" / "descriptions"
    _write_json(
        desc_dir / "all_descriptions.json",
        [
            {"image_index": "001", "scene_pose": [[1, 0, 0, 0]] * 4, "description": "agg-1"},
            {"image_index": "002", "scene_pose": [[1, 0, 0, 0]] * 4, "description": "agg-2"},
        ],
    )
    _write_json(
        desc_dir / "001299.json",
        {"image_index": "001299", "scene_pose": [[1, 0, 0, 0]] * 4, "description": "num"},
    )
    _write_json(
        desc_dir / "frame-000026.json",
        {"image_index": "frame-000026", "scene_pose": [[1, 0, 0, 0]] * 4, "description": "frame"},
    )
    _write_json(desc_dir / "frame-000026_parsed.json", {"parsed_graph": {}})

    frames = load_structured_frame_jsons(desc_dir)
    names = [f.path.name for f in frames]
    assert names == ["001299.json", "frame-000026.json"]


def test_load_structured_frame_jsons_fallback_and_dedupe(tmp_path: Path) -> None:
    desc_dir = tmp_path / "output" / "descriptions"
    _write_json(
        desc_dir / "all_descriptions.json",
        [
            {"image_index": "001", "scene_pose": [[1, 0, 0, 0]] * 4, "description": "a"},
            {"image_index": "001", "scene_pose": [[1, 0, 0, 0]] * 4, "description": "dup"},
            {"image_index": "002", "scene_pose": [[1, 0, 0, 0]] * 4, "description": "b"},
        ],
    )

    frames = load_structured_frame_jsons(desc_dir)
    assert len(frames) == 2
    assert [f.frame["image_index"] for f in frames] == ["001", "002"]


def test_load_parsed_frame_jsons_supports_multiple_prefixes(tmp_path: Path) -> None:
    desc_dir = tmp_path / "output" / "descriptions"
    _write_json(desc_dir / "frame-000026_parsed.json", {"parsed_graph": {"nodes": []}})
    _write_json(desc_dir / "001299_parsed.json", {"parsed_graph": {"nodes": []}})
    _write_json(desc_dir / "foo.json", {"image_index": "foo"})

    frames = load_parsed_frame_jsons(desc_dir)
    assert [f.path.name for f in frames] == ["001299_parsed.json", "frame-000026_parsed.json"]


def test_resolve_raw_frame_path_from_parsed_suffix_only() -> None:
    parsed = Path("/tmp/descriptions/foo_parsed.json")
    assert resolve_raw_frame_path_from_parsed(parsed) == Path("/tmp/descriptions/foo.json")

    with pytest.raises(ValueError):
        resolve_raw_frame_path_from_parsed(Path("/tmp/descriptions/foo.json"))


def test_load_object_labels_and_floor_ids_3rscan(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene-3rscan"
    _write_json(
        scene_dir / "semseg.v2.json",
        {
            "segGroups": [
                {"objectId": 1, "label": "floor"},
                {"objectId": 2, "label": "wood floor"},
                {"objectId": 3, "label": "wall"},
            ]
        },
    )

    labels = load_object_labels(scene_dir, "3rscan")
    floor_ids = load_floor_object_ids(scene_dir, "3rscan")
    assert labels[1] == "floor"
    assert labels[2] == "wood floor"
    assert floor_ids == {1, 2}


def test_load_object_labels_and_floor_ids_scannet(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene0000_00"
    _write_json(
        scene_dir / "scene0000_00.aggregation.json",
        {
            "segGroups": [
                {"objectId": 4, "label": "floor tile", "segments": [1, 2]},
                {"objectId": 5, "label": "chair", "segments": [3]},
            ]
        },
    )

    labels = load_object_labels(scene_dir, "scannet")
    floor_ids = load_floor_object_ids(scene_dir, "scannet")
    assert labels[4] == "floor tile"
    assert 4 in floor_ids
    assert 5 not in floor_ids
