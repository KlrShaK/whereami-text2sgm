#!/usr/bin/env python3
"""
visualize_eval_loc_mk6.py
-------------------------
Evaluate localisation quality using **GPT-parsed description graphs** with
**CLIP embeddings** and **relationship-aware object matching**.

Changes from mk5:
- Scene graphs loaded from ``clip_full_3dssg_graphs.pt`` with ``embedding_type='clip'``
- Query graphs embedded with CLIP (512-dim) instead of word2vec
- ``relationship_aware_topk()`` replaces ``topk_matched_objects()``:
  combines label similarity, attribute similarity, and structural
  relationship compatibility for disambiguation
- Enhanced debug output: GT recall@K, per-component score breakdown
- Centroid recovery uses CLIP embeddings
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Repository imports                                                          #
# --------------------------------------------------------------------------- #

sys.path.append("../data_processing")
sys.path.append("../../../")

from scene_graph import SceneGraph  # noqa: E402
from create_text_embeddings import (  # noqa: E402
    create_embedding_clip,
    create_embeddings_clip_batch,
)
from visualize_3rscan_segments import build_segmented_mesh  # noqa: E402
from localization_dataset_io import (
    load_object_labels,
    load_parsed_frame_jsons as load_parsed_frame_records,
    resolve_raw_frame_path_from_parsed,
)

# --------------------------------------------------------------------------- #
# Import helpers from visualize_loc_prob.py                                   #
# --------------------------------------------------------------------------- #

_THIS_DIR = Path(__file__).resolve().parent
_VIZ = _THIS_DIR / "visualize_loc_prob.py"
if not _VIZ.exists():
    raise FileNotFoundError(f"Expected visualize_loc_prob.py at {_VIZ}")

from importlib.machinery import SourceFileLoader

vizmod = SourceFileLoader("vizlocprob_eval", str(_VIZ)).load_module()  # type: ignore

load_scene = vizmod.load_scene
sample_grid = vizmod.sample_grid
first_hit_is_object = vizmod.first_hit_is_object
colour_objects = vizmod.colour_objects
colormap = vizmod.colormap
dir_to_yaw_pitch = getattr(vizmod, "dir_to_yaw_pitch", None)
best_fov_window = getattr(vizmod, "best_fov_window", None)
average_direction = getattr(vizmod, "average_direction", None)

# --------------------------------------------------------------------------- #
# Import shared utilities from mk4 (reuse everything except graph construction)
# --------------------------------------------------------------------------- #

from visualize_eval_loc_mk4 import (  # noqa: E402
    FrameSelection,
    SceneMetrics,
    build_metrics_table,
    format_args_section,
    camera_center_from_pose,
    compute_metrics,
    compute_view_iou_error,
    select_prediction_point,
    top_n_fov_poses,
    softmax_probs,
    proximity_bonus,
    add_heatmap_markers,
    add_arrow_markers,
    create_camera_frustum,
    _extract_floor_bbox,
    _visible_triangles_from_view,
    _cluster_weighted_prediction,
)


# --------------------------------------------------------------------------- #
# CLIP embedding cache                                                        #
# --------------------------------------------------------------------------- #

_CLIP_CACHE: Dict[str, list] = {}
logger = logging.getLogger(__name__)


def _embed_clip(text: str) -> list:
    """Return a 512-dim CLIP embedding (as Python list) for *text*, cached."""
    key = text.strip().lower()
    cached = _CLIP_CACHE.get(key)
    if cached is not None:
        return cached
    emb = create_embedding_clip(text).tolist()
    _CLIP_CACHE[key] = emb
    return emb


# --------------------------------------------------------------------------- #
# Load scene graphs with CLIP                                                 #
# --------------------------------------------------------------------------- #

def load_scene_graphs_clip(graphs_dir: Path,
                           scene_use_attributes: bool = False) -> Dict[str, SceneGraph]:
    """Load 3DSSG scene graphs from the all-CLIP .pt file."""
    path = graphs_dir / "3dssg" / "clip_full_3dssg_graphs.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"CLIP scene graph file not found: {path}\n"
            "Run reprocess_3dssg_clip.py first to generate it."
        )
    # PyTorch 2.6 defaults torch.load(..., weights_only=True), which breaks
    # loading these trusted local graph pickles.
    g3d = torch.load(path, map_location="cpu", weights_only=False)
    scenes: Dict[str, SceneGraph] = {}
    for sid, graph in g3d.items():
        scenes[sid] = SceneGraph(
            sid,
            graph_type="3dssg",
            graph=graph,
            max_dist=1.0,
            embedding_type="clip",
            use_attributes=scene_use_attributes,
        )
    return scenes


# --------------------------------------------------------------------------- #
# Parsed-frame loading and selection (reused from mk5 pattern)                #
# --------------------------------------------------------------------------- #

def load_parsed_frame_jsons(desc_dir: Path) -> List[FrameSelection]:
    records = load_parsed_frame_records(desc_dir)
    return [FrameSelection(frame=rec.frame, path=rec.path) for rec in records]


def select_parsed_frame(frames: List[FrameSelection],
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
    if policy in ("max_visible", "max_pixels"):
        return max(
            frames,
            key=lambda fs: len(
                fs.frame.get("parsed_graph", {}).get("nodes", [])
            ),
        )
    raise ValueError(f"Unknown frame selection policy '{policy}'")


# --------------------------------------------------------------------------- #
# Centroid recovery using CLIP embeddings                                     #
# --------------------------------------------------------------------------- #

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def recover_centroids(parsed_graph: dict,
                      frame_path: Path,
                      similarity_threshold: float = 0.7) -> Dict[int, dict]:
    """Match parsed node labels to visible_objects via CLIP cosine similarity
    and recover centroid_world for each matched parsed node."""
    try:
        original_path = resolve_raw_frame_path_from_parsed(frame_path)
    except ValueError:
        return {}
    if not original_path.exists():
        return {}

    original = json.loads(original_path.read_text())
    visible_objects = original.get("visible_objects", {}) or {}
    if not visible_objects:
        return {}

    vo_items = list(visible_objects.items())
    vo_labels = [obj.get("label", "") for _, obj in vo_items]
    vo_embeddings = np.array(
        [_embed_clip(lbl) for lbl in vo_labels], dtype=np.float32,
    )
    vo_centroids = [
        np.asarray(obj.get("centroid_world", [0, 0, 0]), dtype=np.float32)
        for _, obj in vo_items
    ]
    vo_ids = [raw_id for raw_id, _ in vo_items]

    nodes = parsed_graph.get("nodes", [])
    meta: Dict[int, dict] = {}

    for node in nodes:
        nid = node["id"]
        label = node.get("label", "")

        if "label_clip" in node and node["label_clip"]:
            node_emb = np.asarray(node["label_clip"], dtype=np.float32)
        else:
            node_emb = np.asarray(_embed_clip(label), dtype=np.float32)

        best_sim = -1.0
        best_vo_idx = -1
        for vi, vo_emb in enumerate(vo_embeddings):
            sim = _cosine_sim(node_emb, vo_emb)
            if sim > best_sim:
                best_sim = sim
                best_vo_idx = vi

        if best_sim >= similarity_threshold and best_vo_idx >= 0:
            meta[nid] = {
                "source_object_id": vo_ids[best_vo_idx],
                "label": label,
                "matched_vo_label": vo_labels[best_vo_idx],
                "match_similarity": best_sim,
                "centroid_world": vo_centroids[best_vo_idx],
            }

    return meta


# --------------------------------------------------------------------------- #
# Build query SceneGraph with CLIP embeddings                                 #
# --------------------------------------------------------------------------- #

def parsed_frame_to_scenegraph_clip(
    parsed_data: dict,
    frame_path: Path,
    centroid_similarity_threshold: float = 0.7,
) -> Tuple[SceneGraph, Dict[int, dict]]:
    """Build a SceneGraph from a pre-parsed frame using CLIP embeddings."""
    parsed_graph = parsed_data.get("parsed_graph", {})
    nodes = parsed_graph.get("nodes", [])
    edges = parsed_graph.get("edges", [])

    meta = recover_centroids(
        parsed_graph, frame_path,
        similarity_threshold=centroid_similarity_threshold,
    )

    # Embed nodes with CLIP
    for node in nodes:
        if "label_clip" not in node or not node["label_clip"]:
            node["label_clip"] = _embed_clip(node["label"])
        if "attributes_clip" not in node:
            node["attributes_clip"] = {
                "all": [_embed_clip(a) for a in node.get("attributes", [])]
            }

    # Embed edges with CLIP
    for edge in edges:
        if "relation_clip" not in edge or not edge["relation_clip"]:
            edge["relation_clip"] = _embed_clip(edge["relationship"])

    graph_dict = {"nodes": nodes, "edges": edges}
    scene_id = parsed_data.get("scene_index", "unknown_scene")
    txt_id = parsed_data.get("source_frame")

    sg = SceneGraph(
        scene_id=scene_id,
        txt_id=txt_id,
        graph_type="scanscribe",
        graph=graph_dict,
        embedding_type="clip",
        use_attributes=True,
    )
    return sg, meta


# --------------------------------------------------------------------------- #
# Relationship-aware Top-K matching                                           #
# --------------------------------------------------------------------------- #

def _get_node_edges(sg: SceneGraph, node_id: int) -> List[Tuple[int, int, np.ndarray]]:
    """Return list of (neighbor_id, edge_index, edge_features) for edges
    connected to *node_id* (either direction)."""
    result = []
    if sg.edge_idx is None or len(sg.edge_idx[0]) == 0:
        return result
    for i, (f, t) in enumerate(zip(sg.edge_idx[0], sg.edge_idx[1])):
        if int(f) == node_id:
            result.append((int(t), i, np.asarray(sg.edge_features[i], dtype=np.float32)))
        elif int(t) == node_id:
            result.append((int(f), i, np.asarray(sg.edge_features[i], dtype=np.float32)))
    return result


def relationship_compatibility(
    q_node_id: int,
    s_node_id: int,
    query_sg: SceneGraph,
    scene_sg: SceneGraph,
    neighbor_w: float = 0.5,
) -> float:
    """Compute relationship compatibility score between query node and scene node.

    For each query edge connected to q_node_id, find the best-matching scene
    edge connected to s_node_id.  Each edge match considers both edge-relation
    similarity and neighbor-label similarity.

    Returns mean of best-match scores (0.0 if no query edges).
    """
    q_edges = _get_node_edges(query_sg, q_node_id)
    if not q_edges:
        return 0.0

    s_edges = _get_node_edges(scene_sg, s_node_id)
    if not s_edges:
        return 0.0

    best_scores: List[float] = []
    for q_neighbor_id, _, q_rel_feat in q_edges:
        q_neighbor = query_sg.nodes.get(q_neighbor_id)
        if q_neighbor is None:
            continue

        q_rel = q_rel_feat / max(np.linalg.norm(q_rel_feat), 1e-9)
        q_neigh_feat = np.asarray(q_neighbor.label_features, dtype=np.float32)
        q_neigh_feat = q_neigh_feat / max(np.linalg.norm(q_neigh_feat), 1e-9)

        best_edge_score = -1.0
        for s_neighbor_id, _, s_rel_feat in s_edges:
            s_neighbor = scene_sg.nodes.get(s_neighbor_id)
            if s_neighbor is None:
                continue

            s_rel = s_rel_feat / max(np.linalg.norm(s_rel_feat), 1e-9)
            s_neigh_feat = np.asarray(s_neighbor.label_features, dtype=np.float32)
            s_neigh_feat = s_neigh_feat / max(np.linalg.norm(s_neigh_feat), 1e-9)

            edge_sim = float(np.dot(q_rel, s_rel))
            neighbor_sim = float(np.dot(q_neigh_feat, s_neigh_feat))
            edge_match = (1 - neighbor_w) * edge_sim + neighbor_w * neighbor_sim

            if edge_match > best_edge_score:
                best_edge_score = edge_match

        if best_edge_score >= 0:
            best_scores.append(best_edge_score)

    if not best_scores:
        return 0.0
    return float(np.mean(best_scores))


def attribute_similarity(q_node, s_node) -> float:
    """Compute attribute similarity between query and scene node.
    Returns 0.0 if either side has no attributes."""
    q_attrs = q_node.attribute_features
    s_attrs = s_node.attribute_features

    if not q_attrs or not s_attrs:
        return 0.0

    q_mean = np.mean(q_attrs, axis=0).astype(np.float32)
    s_mean = np.mean(s_attrs, axis=0).astype(np.float32)

    nq = np.linalg.norm(q_mean)
    ns = np.linalg.norm(s_mean)
    if nq < 1e-9 or ns < 1e-9:
        return 0.0
    return float(np.dot(q_mean, s_mean) / (nq * ns))


def relationship_aware_topk(
    query_sg: SceneGraph,
    scene_sg: SceneGraph,
    k: int = 25,
    return_scores: bool = False,
    dynamic_k: bool = False,
    score_threshold: float = -1.0,
    ensure_query_coverage: bool = False,
    label_weight: float = 0.55,
    attribute_weight: float = 0.15,
    relation_weight: float = 0.3,
    neighbor_weight: float = 0.5,
) -> Tuple[List[int], List[float]] | List[int]:
    """Relationship-aware Top-K matching.

    Combines label cosine similarity, attribute similarity, and structural
    relationship compatibility.

    Same return interface as ``topk_matched_objects``.
    """
    query_obj_count = len(query_sg.nodes)
    if dynamic_k:
        k = query_obj_count
    k = max(1, int(k))

    qids = list(query_sg.nodes)
    sids = list(scene_sg.nodes)
    if not qids or not sids:
        return ([], []) if return_scores else []

    # Pre-compute label similarity matrix
    qmat = np.asarray([query_sg.nodes[qid].label_features for qid in qids], dtype=np.float32)
    smat = np.asarray([scene_sg.nodes[sid].label_features for sid in sids], dtype=np.float32)
    qf = F.normalize(torch.tensor(qmat, dtype=torch.float32), dim=1)
    sf = F.normalize(torch.tensor(smat, dtype=torch.float32), dim=1)

    if qf.numel() == 0 or sf.numel() == 0:
        return ([], []) if return_scores else []

    label_sim = (qf @ sf.T).numpy()  # (|Q|, |S|)

    # Compute combined score matrix
    combined = np.zeros_like(label_sim)
    for qi, qid in enumerate(qids):
        q_node = query_sg.nodes[qid]
        for si, sid in enumerate(sids):
            s_node = scene_sg.nodes[sid]

            l_score = label_sim[qi, si]
            a_score = attribute_similarity(q_node, s_node)
            r_score = relationship_compatibility(
                qid, sid, query_sg, scene_sg, neighbor_w=neighbor_weight
            )

            # Compute effective weights (redistribute if component missing)
            eff_label_w = label_weight
            eff_attr_w = attribute_weight if a_score != 0.0 else 0.0
            eff_rel_w = relation_weight if r_score != 0.0 else 0.0

            total_w = eff_label_w + eff_attr_w + eff_rel_w
            if total_w < 1e-9:
                combined[qi, si] = l_score
                continue

            # Normalize weights to sum to 1
            eff_label_w /= total_w
            eff_attr_w /= total_w
            eff_rel_w /= total_w

            combined[qi, si] = (
                eff_label_w * l_score
                + eff_attr_w * a_score
                + eff_rel_w * r_score
            )

    # Select top-K scene nodes
    picks: List[int] = []
    scores: List[float] = []
    picked_set: set = set()
    S = len(sids)

    if ensure_query_coverage:
        for qi in range(len(qids)):
            row_order = np.argsort(-combined[qi])
            for si in row_order:
                val = combined[qi, si]
                if val < score_threshold:
                    break
                sid = sids[si]
                if sid in picked_set:
                    continue
                picks.append(sid)
                scores.append(float(val))
                picked_set.add(sid)
                break
            if len(picks) == k:
                break

    if len(picks) < k:
        flat = combined.ravel()
        order = np.argsort(-flat)
        for idx in order:
            val = flat[idx]
            if val < score_threshold:
                break
            sid = sids[idx % S]
            if sid in picked_set:
                continue
            picks.append(sid)
            scores.append(float(val))
            picked_set.add(sid)
            if len(picks) == k:
                break

    if return_scores:
        return picks, scores
    return picks


# --------------------------------------------------------------------------- #
# Enhanced debugging with GT comparison                                       #
# --------------------------------------------------------------------------- #

def debug_match_labels_mk6(
    query_sg: SceneGraph,
    scene_sg: SceneGraph,
    frame_path: Path,
    topn: int = 5,
    print_all: bool = False,
    csv_dir: Optional[Path] = None,
    scene_id: str = "",
    frame_id: str = "",
    label_weight: float = 0.55,
    attribute_weight: float = 0.15,
    relation_weight: float = 0.3,
    neighbor_weight: float = 0.5,
    top_k: int = 25,
) -> None:
    """Print per-query-node match details with label/attr/rel score breakdown.

    When the original frame JSON is available, also compute GT recall@K.
    """
    qids = list(query_sg.nodes)
    sids = list(scene_sg.nodes)
    if not qids or not sids:
        logger.debug("    [DEBUG-MK6] No nodes to compare.")
        return

    # Pre-compute label sim
    qmat = np.asarray([query_sg.nodes[qid].label_features for qid in qids], dtype=np.float32)
    smat = np.asarray([scene_sg.nodes[sid].label_features for sid in sids], dtype=np.float32)
    qf = F.normalize(torch.tensor(qmat, dtype=torch.float32), dim=1)
    sf = F.normalize(torch.tensor(smat, dtype=torch.float32), dim=1)
    label_sim = (qf @ sf.T).numpy()

    # Load GT visible_objects if available
    try:
        original_path = resolve_raw_frame_path_from_parsed(frame_path)
    except ValueError:
        original_path = frame_path.with_name(frame_path.stem + ".json")
    gt_labels: Dict[str, str] = {}
    if original_path.exists():
        original = json.loads(original_path.read_text())
        for vo_id, vo in (original.get("visible_objects", {}) or {}).items():
            gt_labels[vo_id] = vo.get("label", "")

    gt_recall_hits = 0
    gt_recall_total = 0

    logger.debug(f"\n    [DEBUG-MK6] Match breakdown for {scene_id} / {frame_id}")
    logger.debug(f"    {'qnode':<20} {'snode':<20} {'label':>6} {'attr':>6} {'rel':>6} {'comb':>7}")
    logger.debug(f"    {'-'*20} {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")

    csv_rows: List[dict] = []

    for qi, qid in enumerate(qids):
        q_node = query_sg.nodes[qid]
        q_label = q_node.label or f"id_{qid}"

        # Compute full scores for this query node
        row_scores: List[Tuple[int, float, float, float, float]] = []
        for si, sid in enumerate(sids):
            s_node = scene_sg.nodes[sid]
            l_score = label_sim[qi, si]
            a_score = attribute_similarity(q_node, s_node)
            r_score = relationship_compatibility(
                qid, sid, query_sg, scene_sg, neighbor_w=neighbor_weight
            )

            eff_label_w = label_weight
            eff_attr_w = attribute_weight if a_score != 0.0 else 0.0
            eff_rel_w = relation_weight if r_score != 0.0 else 0.0
            total_w = eff_label_w + eff_attr_w + eff_rel_w
            if total_w < 1e-9:
                comb = l_score
            else:
                comb = (
                    (eff_label_w / total_w) * l_score
                    + (eff_attr_w / total_w) * a_score
                    + (eff_rel_w / total_w) * r_score
                )

            row_scores.append((si, float(l_score), float(a_score), float(r_score), float(comb)))

            if print_all or csv_dir is not None:
                csv_rows.append({
                    "scene_id": scene_id,
                    "frame_id": frame_id,
                    "query_node": q_label,
                    "query_id": qid,
                    "scene_node": s_node.label or f"id_{sid}",
                    "scene_id_node": sid,
                    "label_sim": f"{l_score:.4f}",
                    "attr_sim": f"{a_score:.4f}",
                    "rel_sim": f"{r_score:.4f}",
                    "combined": f"{comb:.4f}",
                })

        # Sort by combined score
        row_scores.sort(key=lambda x: -x[4])

        # Print top-N
        for rank, (si, l_s, a_s, r_s, c_s) in enumerate(row_scores[:topn]):
            s_label = scene_sg.nodes[sids[si]].label or f"id_{sids[si]}"
            marker = "*" if rank == 0 else " "
            logger.debug(
                f"  {marker} {q_label:<20} {s_label:<20} {l_s:>6.3f} {a_s:>6.3f} {r_s:>6.3f} {c_s:>7.3f}"
            )

        # GT recall check
        if gt_labels:
            gt_recall_total += 1
            # Check if any GT visible_object label matches this query label
            # in the top-K matches
            q_clip = np.asarray(q_node.label_features, dtype=np.float32)
            q_clip_n = q_clip / max(np.linalg.norm(q_clip), 1e-9)
            gt_sims = []
            for vo_id, vo_label in gt_labels.items():
                vo_emb = np.asarray(_embed_clip(vo_label), dtype=np.float32)
                vo_emb_n = vo_emb / max(np.linalg.norm(vo_emb), 1e-9)
                gt_sims.append((vo_id, vo_label, float(np.dot(q_clip_n, vo_emb_n))))
            gt_sims.sort(key=lambda x: -x[2])
            if gt_sims and gt_sims[0][2] > 0.7:
                best_gt_label = gt_sims[0][1]
                # Find rank of best GT match in scene matches
                for rank, (si, _, _, _, _) in enumerate(row_scores):
                    s_label = scene_sg.nodes[sids[si]].label or ""
                    s_emb = np.asarray(scene_sg.nodes[sids[si]].label_features, dtype=np.float32)
                    s_emb_n = s_emb / max(np.linalg.norm(s_emb), 1e-9)
                    gt_emb = np.asarray(_embed_clip(best_gt_label), dtype=np.float32)
                    gt_emb_n = gt_emb / max(np.linalg.norm(gt_emb), 1e-9)
                    if float(np.dot(s_emb_n, gt_emb_n)) > 0.85:
                        if rank < top_k:
                            gt_recall_hits += 1
                        logger.debug(f"      GT: '{best_gt_label}' found at rank {rank}")
                        break

        logger.debug("")

    if gt_recall_total > 0:
        logger.debug(
            f"    [DEBUG-MK6] GT recall@{top_k}: {gt_recall_hits}/{gt_recall_total} "
            f"({100*gt_recall_hits/gt_recall_total:.1f}%)"
        )

    # Save CSV if requested
    if csv_dir is not None and csv_rows:
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"match_scores_{scene_id}_{frame_id}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info(f"    Score matrix saved to {csv_path}")


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #

def ensure_query_root(query_root: Optional[Path], root: Path) -> Path:
    if query_root is not None:
        return query_root
    return root


# --------------------------------------------------------------------------- #
# Main evaluation pipeline                                                    #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate localisation using CLIP embeddings + relationship-aware matching (mk6)."
    )
    parser.add_argument("--root", required=True,
                        help="Root directory containing <scene_id>/ meshes.")
    parser.add_argument("--dataset", required=True, choices=["3rscan", "scannet"],
                        help="Dataset layout used under --root and --query_root.")
    parser.add_argument("--graphs", required=True, type=Path,
                        help="processed_data directory holding 3dssg/*.pt files.")
    parser.add_argument("--query_root", type=Path,
                        help="Root containing per-scene output/descriptions/*_parsed.json")

    parser.add_argument("--scene_ids", nargs="+",
                        help="Subset of scene IDs to evaluate.")
    parser.add_argument("--max_scenes", type=int,
                        help="Limit number of scenes processed.")
    parser.add_argument("--visualize_scene",
                        help="Scene ID to focus on for visualisation.")

    parser.add_argument("--frame_policy",
                        choices=["first", "index", "random", "max_visible", "max_pixels", "all"],
                        default="max_visible",
                        help="Strategy to pick parsed frame JSON(s) per scene. "
                             "'all' evaluates every parsed frame in the scene.")
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--dynamic_top_k", action="store_true")
    parser.add_argument("--score_threshold", type=float, default=-1.0)
    parser.add_argument("--ensure_query_coverage", dest="ensure_query_coverage",
                        action="store_true", default=True)
    parser.add_argument("--no_ensure_query_coverage", dest="ensure_query_coverage",
                        action="store_false")
    parser.add_argument("--use_subgraph", action="store_true")
    parser.add_argument("--scene_use_attributes", action="store_true")
    parser.add_argument("--centroid_similarity_threshold", type=float, default=0.7,
                        help="Minimum CLIP cosine similarity for parsed-node to visible-object matching.")

    # Relationship-aware matching weights
    parser.add_argument("--relation_weight", type=float, default=0.3,
                        help="Weight of relationship score in combined matching (default: 0.3).")
    parser.add_argument("--attribute_weight", type=float, default=0.15,
                        help="Weight of attribute score in combined matching (default: 0.15).")
    parser.add_argument("--neighbor_weight", type=float, default=0.5,
                        help="Within rel_score, weight of neighbor-label similarity vs edge-relation similarity (default: 0.5).")
    # Note: label_weight = 1 - relation_weight - attribute_weight

    parser.add_argument("--no_homogenize_label_embeddings", dest="homogenize_label_embeddings",
                        action="store_false", default=False,
                        help="Use CLIP features directly (default).")
    parser.add_argument("--homogenize_label_embeddings", dest="homogenize_label_embeddings",
                        action="store_true")

    parser.add_argument("--debug_match_labels", action="store_true")
    parser.add_argument("--debug_match_all_scores", action="store_true")
    parser.add_argument("--debug_match_topn", type=int, default=5)
    parser.add_argument("--debug_match_csv_dir", type=Path)

    parser.add_argument("--grid_step", type=float, default=0.25)
    parser.add_argument("--eye_height", type=float, default=1.6)
    parser.add_argument("--prob_eps", type=float, default=1e-6)
    parser.add_argument("--hit_radii", nargs="+", type=float,
                        default=[0.75, 1.0, 1.5, 2.0, 2.5])
    parser.add_argument("--mass_percentiles", nargs="+", type=float,
                        default=[50.0, 90.0])
    parser.add_argument("--top_k_min_dist", type=int, default=10)
    parser.add_argument("--prediction_strategy",
                        choices=["argmax", "random", "weighted"], default="weighted")
    parser.add_argument("--cluster_bandwidth", type=float, default=1.0)
    parser.add_argument("--max_cluster_points", type=int, default=50)

    parser.add_argument("--show_heatmap", action="store_true")
    parser.add_argument("--show_3d", action="store_true")
    parser.add_argument("--show_arrows", action="store_true")
    parser.add_argument("--h_fov_deg", type=float, default=39.31)
    parser.add_argument("--v_fov_deg", type=float, default=64.76)
    parser.add_argument("--arrow_stride", type=int, default=2)
    parser.add_argument("--arrow_len", type=float, default=0.0)
    parser.add_argument("--score_tau", type=float, default=1.5)
    parser.add_argument("--distance_bonus_weight", type=float, default=0.5)
    parser.add_argument("--distance_bonus_decay", type=float, default=2.0)

    parser.add_argument("--save_metrics", type=Path)
    parser.add_argument("--log_file", type=Path, default=Path("eval_loc_summary_mk6.log"))
    parser.add_argument("--top_pose_count", type=int, default=5)
    parser.add_argument("--resume", nargs="?", const="__save_metrics__", default=None,
                        help="Resume from a prior checkpoint JSON. If a path is given, "
                             "use that file; if --resume is passed without a path, "
                             "defaults to the --save_metrics path.")
    parser.add_argument("--checkpoint_interval", type=int, default=200,
                        help="Save a checkpoint every N scenes (requires --save_metrics).")
    return parser.parse_args()


def evaluate_scene(scene_id: str,
                   scene_graph: SceneGraph,
                   args: argparse.Namespace,
                   rng: np.random.Generator) -> List[SceneMetrics]:
    mesh_root = Path(args.root)
    scene_dir = mesh_root / scene_id
    if not scene_dir.exists():
        logger.warning(f"Scene directory missing for {scene_id} - skipped.")
        return []

    query_root = ensure_query_root(args.query_root, Path(args.root))
    desc_dir = query_root / scene_id / "output" / "descriptions"
    if not desc_dir.exists():
        desc_dir = scene_dir / "output" / "descriptions"

    frames = load_parsed_frame_jsons(desc_dir)
    if not frames:
        logger.warning(f"No parsed frame JSONs under {desc_dir} - skipped.")
        return []

    if args.frame_policy == "all":
        selected_frames = frames
    else:
        selection = select_parsed_frame(frames, args.frame_policy, args.frame_index, rng)
        if selection is None:
            logger.warning(f"Frame selection failed for {scene_id} - skipped.")
            return []
        selected_frames = [selection]

    if args.frame_policy == "all" and len(selected_frames) > 1:
        logger.info(f"    evaluating all {len(selected_frames)} parsed frames in scene")

    metrics_items: List[SceneMetrics] = []
    for frame_idx, selection in enumerate(selected_frames, start=1):
        if args.frame_policy == "all" and len(selected_frames) > 1:
            frame_id = str(selection.frame.get("source_frame", selection.path.name))
            logger.info(f"    frame [{frame_idx:03d}/{len(selected_frames):03d}]: "
                        f"{frame_id} ({selection.path.name})")
        metric = _evaluate_selected_frame(scene_id, scene_graph, selection, scene_dir, args, rng)
        if metric is not None:
            metrics_items.append(metric)

    return metrics_items


def _evaluate_selected_frame(scene_id: str,
                             scene_graph: SceneGraph,
                             selection: FrameSelection,
                             scene_dir: Path,
                             args: argparse.Namespace,
                             rng: np.random.Generator) -> Optional[SceneMetrics]:
    frame_data = selection.frame

    try:
        caption_graph, caption_meta = parsed_frame_to_scenegraph_clip(
            frame_data,
            selection.path,
            centroid_similarity_threshold=args.centroid_similarity_threshold,
        )
    except Exception as exc:
        logger.warning(f"Failed to build caption graph for {scene_id}: {exc}")
        return None

    if not caption_meta:
        logger.warning(f"{scene_id}: no parsed nodes matched visible objects - skipped.")
        return None

    frame_id_dbg = str(frame_data.get("source_frame", selection.path.name))

    # Print centroid recovery summary
    n_nodes = len(frame_data.get("parsed_graph", {}).get("nodes", []))
    logger.debug(
        f"    parsed graph: {n_nodes} nodes, {len(caption_meta)} matched to visible objects"
    )
    for nid, m in sorted(caption_meta.items()):
        logger.debug(
            f"      node[{nid}] '{m['label']}' -> vo '{m['matched_vo_label']}' "
            f"(sim={m['match_similarity']:.3f})"
        )

    # Compute matching weights
    label_w = 1.0 - args.relation_weight - args.attribute_weight

    if args.debug_match_labels or args.debug_match_all_scores or args.debug_match_csv_dir is not None:
        debug_match_labels_mk6(
            caption_graph,
            scene_graph,
            selection.path,
            topn=args.debug_match_topn,
            print_all=args.debug_match_all_scores,
            csv_dir=args.debug_match_csv_dir,
            scene_id=scene_id,
            frame_id=frame_id_dbg,
            label_weight=label_w,
            attribute_weight=args.attribute_weight,
            relation_weight=args.relation_weight,
            neighbor_weight=args.neighbor_weight,
            top_k=args.top_k,
        )

    # Load GT pose from original frame JSON
    try:
        original_path = resolve_raw_frame_path_from_parsed(selection.path)
    except ValueError:
        logger.warning(f"Invalid parsed frame path: {selection.path} - skipped.")
        return None
    if not original_path.exists():
        logger.warning(f"Original frame JSON not found: {original_path} - skipped.")
        return None

    original_frame = json.loads(original_path.read_text())
    gt_pose = original_frame.get("scene_pose")
    if gt_pose is None:
        logger.warning(f"scene_pose missing in {original_path} - skipped.")
        return None

    pose_mat = np.asarray(gt_pose, dtype=np.float64)
    gt_cam = camera_center_from_pose(pose_mat)
    rot_cam_world = pose_mat[:3, :3]
    forward_cv = rot_cam_world @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward_o3d = forward_cv
    norm_forward = np.linalg.norm(forward_o3d)
    gt_dir = forward_o3d / norm_forward if norm_forward > 1e-6 else None

    # Relationship-aware matching
    obj_ids, obj_scores = relationship_aware_topk(
        caption_graph,
        scene_graph,
        k=args.top_k,
        return_scores=True,
        dynamic_k=args.dynamic_top_k,
        score_threshold=args.score_threshold,
        ensure_query_coverage=args.ensure_query_coverage,
        label_weight=label_w,
        attribute_weight=args.attribute_weight,
        relation_weight=args.relation_weight,
        neighbor_weight=args.neighbor_weight,
    )
    if not obj_ids:
        logger.warning(f"{scene_id}: no matches - skipped.")
        return None

    mesh, tri2obj, obj2faces = load_scene(scene_dir, dataset=args.dataset)
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
    floor_bbox = _extract_floor_bbox(scene_dir, verts, tris, obj2faces, args.dataset)
    if floor_bbox is not None:
        x_min, x_max = floor_bbox["x_min"], floor_bbox["x_max"]
        y_min, y_max = floor_bbox["y_min"], floor_bbox["y_max"]
        z_eye = floor_bbox["z_max"] + args.eye_height
        logger.info(
            f"    floor bbox: X=[{x_min:.2f}, {x_max:.2f}] "
            f"Y=[{y_min:.2f}, {y_max:.2f}] Z_eye={z_eye:.2f} m"
        )
    else:
        logger.warning("    No floor in semantic labels - sampling over full mesh bounds.")
        x_min, x_max = float(verts[:, 0].min()), float(verts[:, 0].max())
        y_min, y_max = float(verts[:, 1].min()), float(verts[:, 1].max())
        z_eye = float(verts[:, 2].min()) + args.eye_height

    gx = np.arange(x_min, x_max + 1e-4, args.grid_step)
    gy = np.arange(y_min, y_max + 1e-4, args.grid_step)
    Nx, Ny = len(gx), len(gy)
    xv, yv = np.meshgrid(gx, gy, indexing="xy")
    n = xv.size
    cams = np.stack([xv.ravel(), yv.ravel(), np.full(n, z_eye)], axis=1)

    centroids: Dict[int, np.ndarray] = {}
    for oid in obj_ids:
        faces = obj2faces.get(int(oid))
        if faces is None or not len(faces):
            continue
        centroids[int(oid)] = verts[np.unique(tris[faces].ravel())].mean(axis=0)

    if not centroids:
        logger.warning(f"{scene_id}: matched objects missing geometry - skipped.")
        return None

    visible_dirs: List[List[np.ndarray]] = [[] for _ in range(len(cams))]
    visible_dists: List[List[float]] = [[] for _ in range(len(cams))]
    for idx, cam in enumerate(cams):
        for oid, centre in centroids.items():
            if first_hit_is_object(cam, centre, oid, rc, tri2obj):
                d = centre - cam
                l = np.linalg.norm(d)
                if l > 1e-6:
                    visible_dirs[idx].append(d / l)
                    visible_dists[idx].append(float(l))

    counts = np.array([len(v) for v in visible_dirs], dtype=np.int32)
    total = counts.sum()
    if total == 0:
        logger.warning(f"{scene_id}: matched objects invisible from grid - skipped.")
        return None

    dist_bonus = np.array(
        [proximity_bonus(np.asarray(d, dtype=np.float64), args.distance_bonus_decay)
         for d in visible_dists],
        dtype=np.float64,
    )
    grid_scores = counts.astype(np.float64) + args.distance_bonus_weight * dist_bonus
    grid_probs = softmax_probs(grid_scores, args.score_tau)

    pred_idx, metrics = compute_metrics(cams, grid_probs, gt_cam,
                                        hit_radii=args.hit_radii,
                                        mass_percentiles=args.mass_percentiles,
                                        topk_k=args.top_k_min_dist)
    metrics.scene_id = scene_id
    metrics.frame_id = frame_id_dbg
    metrics.matched_objects = len(obj_ids)

    try:
        pred_cam_grid, grid_sel_idx, grid_sel_weights = select_prediction_point(
            cams, grid_probs,
            strategy=args.prediction_strategy,
            rng=rng,
            bandwidth=args.cluster_bandwidth,
            max_points=args.max_cluster_points,
        )
    except ValueError:
        pred_cam_grid = cams[pred_idx]
        grid_sel_idx = [int(pred_idx)]
        grid_sel_weights = np.asarray([1.0], dtype=np.float64)

    hfov_rad = math.radians(args.h_fov_deg)
    vfov_rad = math.radians(args.v_fov_deg)

    # Arrow-based aggregation
    arrow_positions: List[np.ndarray] = []
    arrow_dirs: List[np.ndarray] = []
    arrow_counts: List[float] = []

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
                    yaw, pit = dir_to_yaw_pitch(vec)
                    yaws[i] = yaw
                    pits[i] = pit
                sel, count = best_fov_window(yaws, pits, hfov_rad, vfov_rad)
                if count == 0:
                    continue
                mdir = average_direction(dirs, sel)
                if mdir is None:
                    continue
                local_dists = np.asarray(visible_dists[idx], dtype=np.float64)
                sel_bonus = proximity_bonus(local_dists[sel], args.distance_bonus_decay)
                arrow_score = float(count) + args.distance_bonus_weight * sel_bonus
                arrow_positions.append(cams[idx])
                arrow_dirs.append(mdir)
                arrow_counts.append(arrow_score)

    arrow_probs: Optional[np.ndarray] = None
    pred_cam_arrow: Optional[np.ndarray] = None
    pred_dir_arrow: Optional[np.ndarray] = None
    arrow_sel_idx: List[int] = []
    arrow_sel_weights = np.asarray([], dtype=np.float64)

    if arrow_counts:
        arrow_positions_np = np.asarray(arrow_positions, dtype=np.float64)
        arrow_dirs_np = np.asarray(arrow_dirs, dtype=np.float64)
        arrow_probs = softmax_probs(np.asarray(arrow_counts, dtype=np.float64),
                                    args.score_tau)
        top_fov_poses = top_n_fov_poses(arrow_positions_np, arrow_probs,
                                        n=args.top_pose_count, rng=rng,
                                        directions=arrow_dirs_np)
        try:
            pred_cam_arrow, arrow_sel_idx, arrow_sel_weights = select_prediction_point(
                arrow_positions_np, arrow_probs,
                strategy=args.prediction_strategy,
                rng=rng,
                bandwidth=args.cluster_bandwidth,
                max_points=args.max_cluster_points,
            )
        except ValueError:
            idx_fallback = int(np.argmax(arrow_probs))
            pred_cam_arrow = arrow_positions_np[idx_fallback]
            arrow_sel_idx = [idx_fallback]
            arrow_sel_weights = np.asarray([1.0], dtype=np.float64)

        if arrow_sel_idx:
            dir_vectors = arrow_dirs_np[arrow_sel_idx]
            weight_vec = arrow_sel_weights
            if weight_vec.shape[0] != len(arrow_sel_idx):
                weight_vec = np.ones(len(arrow_sel_idx), dtype=np.float64)
            weight_vec = np.clip(weight_vec, 0.0, None)
            if not np.any(weight_vec > 0):
                weight_vec = np.ones_like(weight_vec)
            weight_vec /= weight_vec.sum()
            mean_dir = np.sum(weight_vec[:, None] * dir_vectors, axis=0)
            norm_dir = float(np.linalg.norm(mean_dir))
            if norm_dir > 1e-6:
                pred_dir_arrow = mean_dir / norm_dir

    pred_source_primary = "arrow_field" if pred_cam_arrow is not None else "grid_probability"
    pred_cam_primary = pred_cam_arrow if pred_cam_arrow is not None else pred_cam_grid
    pred_dir_primary = pred_dir_arrow if pred_cam_arrow is not None else None

    pred_source = f"{pred_source_primary}:{args.prediction_strategy}"
    metrics.distance_error = float(np.linalg.norm(pred_cam_primary - gt_cam))
    if gt_dir is not None and pred_dir_primary is not None:
        dot = float(np.clip(np.dot(gt_dir, pred_dir_primary), -1.0, 1.0))
        metrics.angular_error_deg = float(math.degrees(math.acos(dot)))
    else:
        metrics.angular_error_deg = None

    grid_err = float(np.linalg.norm(pred_cam_grid - gt_cam))
    logger.info(
        f"    predicted camera (grid:{args.prediction_strategy}): "
        f"{pred_cam_grid.tolist()} | err={grid_err:.3f} m"
    )
    if pred_cam_arrow is not None:
        arrow_err = float(np.linalg.norm(pred_cam_arrow - gt_cam))
        logger.info(
            f"    predicted camera (arrow:{args.prediction_strategy}): "
            f"{pred_cam_arrow.tolist()} | err={arrow_err:.3f} m"
        )
        if pred_dir_arrow is not None:
            logger.info(f"    approx. viewing direction (arrow): {pred_dir_arrow.tolist()}")
        else:
            logger.info("    approx. viewing direction (arrow): n/a (no directional vote)")
    else:
        logger.info("    predicted camera (arrow): n/a (no valid FOV windows)")
    logger.info(f"    primary prediction used for metrics: {pred_source}")

    # Per-object visibility debug
    obj_labels = load_object_labels(scene_dir, args.dataset)

    gt_vis_oids: List[int] = []
    pred_vis_oids: List[int] = []
    for oid, centre in centroids.items():
        if first_hit_is_object(gt_cam, centre, oid, rc, tri2obj):
            gt_vis_oids.append(oid)
        if first_hit_is_object(pred_cam_primary, centre, oid, rc, tri2obj):
            pred_vis_oids.append(oid)

    gt_vis_oid_set = set(gt_vis_oids)
    pred_vis_oid_set = set(pred_vis_oids)
    missed = gt_vis_oid_set - pred_vis_oid_set
    score_map = {int(oid): float(score) for oid, score in zip(obj_ids, obj_scores)}
    logger.debug(f"    [DEBUG] matched-object visibility  ({len(centroids)} objects):")
    logger.debug(f"    {'oid':<8} {'label':<22} {'score':>7} {'d_GT':>7} {'GT':>4} {'d_pred':>8} {'pred':>5}")
    for oid in sorted(centroids):
        centre = centroids[oid]
        label = obj_labels.get(oid, "?")
        score = score_map.get(oid, float("nan"))
        d_gt   = float(np.linalg.norm(centre - gt_cam))
        d_pred = float(np.linalg.norm(centre - pred_cam_primary))
        v_gt   = "YES" if oid in gt_vis_oid_set   else "no"
        v_pred = "YES" if oid in pred_vis_oid_set else "no"
        logger.debug(f"    {oid:<8} {label:<22} {score:>7.3f} {d_gt:>6.2f}m {v_gt:>4} {d_pred:>7.2f}m {v_pred:>5}")
    shared = gt_vis_oid_set & pred_vis_oid_set

    logger.debug(
        f"    [DEBUG] GT sees {len(gt_vis_oids)}/{len(centroids)} | "
        f"pred sees {len(pred_vis_oids)}/{len(centroids)} | "
        f"shared {len(shared)} | missed by pred: {len(missed)}"
    )
    if missed:
        logger.debug(f"    {'oid':<8} {'label':<22} {'d_GT':>7} {'d_pred':>8}")
        for oid in sorted(missed):
            centre = centroids[oid]
            label  = obj_labels.get(oid, "?")
            d_gt   = float(np.linalg.norm(centre - gt_cam))
            d_pred = float(np.linalg.norm(centre - pred_cam_primary))
            logger.debug(f"    {oid:<8} {label:<22} {d_gt:>6.2f}m {d_pred:>7.2f}m")

    iou_val, iou_err, gt_vis_set, pred_vis_set = compute_view_iou_error(
        gt_cam, gt_dir,
        pred_cam_primary, pred_dir_primary,
        hfov=hfov_rad, vfov=vfov_rad,
        rc=rc, geom_id=int(mesh_id),
        tri_pts=tri_pts, tri_centroids=tri_centroids, tri_areas=tri_areas,
        near=0.05, far=None,
    )
    metrics.iou_error = iou_err
    if iou_val is not None and iou_err is not None:
        logger.info(f"    view IoU: {iou_val:.3f} | IoU error: {iou_err:.3f}")
    else:
        logger.info("    view IoU: n/a (missing direction or empty visibility)")

    # --- Visualisation (same as mk5) ---
    if args.show_heatmap:
        plt.figure(figsize=(6.5, 6.2))
        sc = plt.scatter(cams[:, 0], cams[:, 1], c=grid_probs,
                         cmap="viridis", s=14)
        plt.colorbar(sc, label="Probability")
        plt.axis("equal")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.title(f"{scene_id} · {metrics.frame_id} · grid {args.grid_step:.2f} m (mk6-CLIP)")
        add_heatmap_markers(gt_cam,
                            pred_grid=pred_cam_grid,
                            pred_arrow=pred_cam_arrow,
                            label_grid=f"Pred grid ({args.prediction_strategy})",
                            label_arrow=f"Pred arrow ({args.prediction_strategy})")
        plt.tight_layout()
        plt.show()

    if args.show_arrows:
        if arrow_probs is not None and arrow_positions:
            max_len = (0.9 * args.grid_step) if args.arrow_len <= 0 else args.arrow_len
            W_np = np.asarray(arrow_probs, dtype=np.float32)
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
            plt.colorbar(label="FOV probability")
            plt.axis("equal")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.title(f"{scene_id} · {metrics.frame_id} · FOV arrows "
                      f"(H={math.degrees(hfov_rad):.0f}°, V={math.degrees(vfov_rad):.0f}°) (mk6)")
            add_arrow_markers(gt_cam,
                              pred_grid=pred_cam_grid,
                              pred_arrow=pred_cam_arrow)
            plt.tight_layout()
            plt.show()
        else:
            logger.info("    Arrow plot skipped (no valid FOV windows).")

    if args.show_3d:
        matched_set: set[int] = {int(o) for o in obj_ids}
        frustum_scale = max(args.grid_step * 3.0, 0.6)
        try:
            mesh_vis, obj_stats = build_segmented_mesh(scene_dir, seed=42, dataset=args.dataset)
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
        except Exception as exc:
            logger.warning(f"    Segment mesh loading failed ({exc}) - falling back to legacy mesh.")
            mesh_vis = colour_objects(mesh, obj2faces, obj_ids)
            obj_stats = []
        if not mesh_vis.has_vertex_normals():
            mesh_vis.compute_vertex_normals()

        from open3d.visualization import gui, rendering

        global GUI_INITIALISED
        if not GUI_INITIALISED:
            gui.Application.instance.initialize()
            GUI_INITIALISED = True

        vis = o3d.visualization.O3DVisualizer(f"{scene_id} – mk6 localisation eval", 1280, 800)
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

        prob_material = rendering.MaterialRecord()
        prob_material.shader = "defaultLit"
        prob_material.base_color = [1.0, 1.0, 1.0, 1.0]
        for idx_point, (point, colour) in enumerate(zip(cams, colormap(grid_probs))):
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

        pred_sphere_grid = o3d.geometry.TriangleMesh.create_sphere(radius=0.085)
        pred_sphere_grid.translate(pred_cam_grid)
        pred_sphere_grid.paint_uniform_color([1.0, 0.9, 0.0])
        if not pred_sphere_grid.has_vertex_normals():
            pred_sphere_grid.compute_vertex_normals()
        vis.add_geometry("pred_cam_grid", pred_sphere_grid, material)
        vis.add_3d_label(pred_cam_grid, f"Pred grid ({args.prediction_strategy})")

        pred_sphere_arrow = None
        if pred_cam_arrow is not None:
            pred_sphere_arrow = o3d.geometry.TriangleMesh.create_sphere(radius=0.082)
            pred_sphere_arrow.translate(pred_cam_arrow)
            pred_sphere_arrow.paint_uniform_color([0.1, 0.8, 0.9])
            if not pred_sphere_arrow.has_vertex_normals():
                pred_sphere_arrow.compute_vertex_normals()
            vis.add_geometry("pred_cam_arrow", pred_sphere_arrow, material)
            vis.add_3d_label(pred_cam_arrow, f"Pred arrow ({args.prediction_strategy})")

        pred_sphere_primary = pred_sphere_arrow if pred_sphere_arrow is not None else pred_sphere_grid

        frustum_mat = None
        frustum_mat_pred = None
        frustum_gt = create_camera_frustum(gt_cam, gt_dir,
                                           colour=(1.0, 0.0, 0.0),
                                           h_fov=hfov_rad, v_fov=vfov_rad,
                                           scale=frustum_scale)
        pred_colour = (0.1, 0.8, 0.9) if pred_cam_arrow is not None else (1.0, 0.9, 0.0)
        frustum_pred = create_camera_frustum(pred_cam_primary, pred_dir_primary,
                                             colour=pred_colour,
                                             h_fov=hfov_rad, v_fov=vfov_rad,
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

        # IoU overlay window
        vis_iou = o3d.visualization.O3DVisualizer(f"{scene_id} – IoU overlap (mk6)", 1280, 800)
        vis_iou.show_settings = False
        base_mat = rendering.MaterialRecord()
        base_mat.shader = "defaultLitTransparency"
        base_mat.base_color = [0.8, 0.8, 0.8, 0.18]
        vis_iou.add_geometry("mesh_base", mesh, base_mat)

        def _subset_mesh(base: o3d.geometry.TriangleMesh,
                         tris_idx: set[int],
                         colour: Tuple[float, float, float],
                         alpha: float) -> Optional[o3d.geometry.TriangleMesh]:
            if not tris_idx:
                return None
            idx_arr = np.asarray(sorted(tris_idx), dtype=np.int64)
            verts_arr = np.asarray(base.vertices)
            tris_arr = np.asarray(base.triangles)[idx_arr]
            uniq, inv = np.unique(tris_arr.reshape(-1), return_inverse=True)
            new_verts = verts_arr[uniq]
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
            ("iou_gt_only", gt_only, (1.0, 0.0, 0.0), 0.65),
            ("iou_pred_only", pred_only, (1.0, 0.85, 0.0), 0.65),
            ("iou_both", both, (1.0, 0.4, 0.0), 0.85),
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
        vis_iou.add_geometry("pred_cam_iou", pred_sphere_primary, material)
        if frustum_gt is not None:
            vis_iou.add_geometry("frustum_gt_iou", frustum_gt, frustum_mat)
        if frustum_pred is not None:
            vis_iou.add_geometry("frustum_pred_iou", frustum_pred, frustum_mat_pred)
        vis_iou.reset_camera_to_default()
        gui.Application.instance.add_window(vis_iou)
        gui.Application.instance.run()

    return metrics


def _metric_to_dict(m: SceneMetrics) -> dict:
    """Serialize a SceneMetrics to a JSON-compatible dict."""
    return {
        "scene_id": m.scene_id,
        "frame_id": m.frame_id,
        "hit_masses": {str(k): v for k, v in m.hit_masses.items()},
        "mass_radii": {str(k): v for k, v in m.mass_radii.items()},
        "topk_min_dist": m.topk_min_dist,
        "distance_error": m.distance_error,
        "angular_error_deg": m.angular_error_deg,
        "iou_error": m.iou_error,
        "grid_points": m.grid_points,
        "matched_objects": m.matched_objects,
    }


def _save_checkpoint(path: Path, metrics: List[SceneMetrics]) -> None:
    """Write an intermediate checkpoint with metrics collected so far."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "metrics": [_metric_to_dict(m) for m in metrics],
    }, indent=2))
    logger.info(f"    [checkpoint] {len(metrics)} metrics saved to {path}")


def _metrics_from_dict(d: dict) -> SceneMetrics:
    """Reconstruct a SceneMetrics from a saved JSON dict."""
    return SceneMetrics(
        scene_id=d["scene_id"],
        frame_id=d["frame_id"],
        hit_masses={float(k): v for k, v in d["hit_masses"].items()},
        mass_radii={float(k): v for k, v in d["mass_radii"].items()},
        topk_min_dist=d["topk_min_dist"],
        distance_error=d["distance_error"],
        angular_error_deg=d.get("angular_error_deg"),
        grid_points=d["grid_points"],
        matched_objects=d["matched_objects"],
        iou_error=d.get("iou_error"),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    args.hit_radii = [float(r) for r in args.hit_radii]
    args.mass_percentiles = [float(p) for p in args.mass_percentiles]
    params_text = format_args_section(args)
    rng = np.random.default_rng(seed=args.seed)
    logger.info(f"Dataset mode: {args.dataset}")

    scenes = load_scene_graphs_clip(args.graphs, scene_use_attributes=args.scene_use_attributes)

    candidate_ids = list(scenes.keys())
    if args.visualize_scene:
        if args.scene_ids:
            logger.warning("--visualize_scene overrides --scene_ids.")
        if args.visualize_scene not in scenes:
            logger.error(f"Requested scene '{args.visualize_scene}' not found in processed graphs.")
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

    prior_metrics: List[SceneMetrics] = []
    if args.resume is not None:
        if args.resume == "__save_metrics__":
            if args.save_metrics is None:
                logger.error("--resume without a path requires --save_metrics to be set.")
                return
            args.resume = args.save_metrics
        else:
            args.resume = Path(args.resume)
        if not args.resume.exists():
            logger.info(f"Resume file {args.resume} not found — starting fresh.")
            args.resume = None
    if args.resume is not None:
        prior_data = json.loads(args.resume.read_text())
        all_prior_metrics = [_metrics_from_dict(d) for d in prior_data.get("metrics", [])]
        prior_scene_ids = list(dict.fromkeys(m.scene_id for m in all_prior_metrics))
        if prior_scene_ids:
            last_scene = prior_scene_ids[-1]
            keep_ids = set(prior_scene_ids[:-1])
            prior_metrics = [m for m in all_prior_metrics if m.scene_id in keep_ids]
            before = len(candidate_ids)
            candidate_ids = [sid for sid in candidate_ids if sid not in keep_ids]
            logger.info(f"Resuming: kept {len(prior_metrics)} results from "
                        f"{len(keep_ids)} completed scenes, "
                        f"re-evaluating from '{last_scene}', "
                        f"{len(candidate_ids)} remaining")

    label_w = 1.0 - args.relation_weight - args.attribute_weight
    logger.info(f"Evaluating {len(candidate_ids)} scene(s) [mk6 - CLIP + relationship-aware matching]")
    logger.info(
        f"  weights: label={label_w:.2f}, attribute={args.attribute_weight:.2f}, "
        f"relation={args.relation_weight:.2f}, neighbor={args.neighbor_weight:.2f}"
    )

    metrics_list: List[SceneMetrics] = []
    for idx, sid in enumerate(candidate_ids, start=1):
        logger.info(f"[{idx:03d}/{len(candidate_ids):03d}] {sid}")
        scene_metrics_items = evaluate_scene(sid, scenes[sid], args, rng)
        if not scene_metrics_items:
            continue
        metrics_list.extend(scene_metrics_items)
        for scene_metrics in scene_metrics_items:
            logger.info(f"    frame: {scene_metrics.frame_id}")
            logger.info(f"    matches: {scene_metrics.matched_objects} | grid pts: {scene_metrics.grid_points}")
            hit_line = " | ".join(
                f"hit@{r:.2f}m: {scene_metrics.hit_masses.get(r, 0.0):.3f}"
                for r in sorted(scene_metrics.hit_masses)
            )
            logger.info(f"    {hit_line}")
            mass_line = " | ".join(
                f"R{p:.0f}%: {scene_metrics.mass_radii.get(p, float('nan')):.3f} m"
                for p in sorted(scene_metrics.mass_radii)
            )
            if mass_line:
                logger.info(f"    mass-radius: {mass_line}")
            ang_err = ("n/a" if scene_metrics.angular_error_deg is None
                       else f"{scene_metrics.angular_error_deg:.2f}°")
            logger.info(
                f"    topK{args.top_k_min_dist} min dist: {scene_metrics.topk_min_dist:.3f} m | "
                f"dist_err: {scene_metrics.distance_error:.3f} m | ang_err: {ang_err}"
            )
            if scene_metrics.iou_error is not None:
                logger.info(f"    view IoU error: {scene_metrics.iou_error:.3f}")
        if (args.save_metrics
                and args.checkpoint_interval > 0
                and idx % args.checkpoint_interval == 0):
            _save_checkpoint(args.save_metrics, prior_metrics + metrics_list)

    metrics_list = prior_metrics + metrics_list

    if not metrics_list:
        logger.warning("No scenes produced metrics. Nothing to report.")
        if args.log_file:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            payload = "No scenes produced metrics.\n\n" + params_text + "\n"
            args.log_file.write_text(payload)
            logger.info(f"Empty summary logged to {args.log_file}")
        return

    table_text = build_metrics_table(metrics_list,
                                     args.hit_radii,
                                     args.mass_percentiles,
                                     args.top_k_min_dist)
    if table_text:
        logger.info("Scene-level summary table -------------------------------")
        logger.info("\n%s", table_text)
        logger.info("---------------------------------------------------------")

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
        "Aggregate metrics (mk6 — CLIP + relationship-aware) ------",
        f"  TopK{args.top_k_min_dist} min dist (m): mean={mean_topk:.3f} | median={med_topk:.3f}",
        f"  Distance error (m)      : mean={mean_err:.3f} | median={med_err:.3f}",
        "---------------------------------------------------------\n",
    ]
    for r in sorted(hit_stats):
        mean_hit, med_hit = hit_stats[r]
        agg_lines.insert(-1,
                         f"  Hit@{r:.2f}m              : mean={mean_hit:.3f} | median={med_hit:.3f}")
    for p in sorted(mass_radius_stats):
        mean_r, med_r = mass_radius_stats[p]
        agg_lines.insert(-1,
                         f"  Mass-radius R{p:.0f}% (m): mean={mean_r:.3f} | median={med_r:.3f}")
    if mean_ang is not None and med_ang is not None:
        agg_lines.insert(-1,
                         f"  Angular error (deg)   : mean={mean_ang:.2f} | median={med_ang:.2f}")
    if mean_iou_err is not None and med_iou_err is not None:
        agg_lines.insert(-1,
                         f"  View IoU error     : mean={mean_iou_err:.3f} | median={med_iou_err:.3f}")
    logger.info("\n%s", "\n".join(agg_lines))

    log_sections: List[str] = [params_text]
    if table_text:
        log_sections.append("Scene-level summary table")
        log_sections.append(table_text)
    log_sections.append("\n".join(agg_lines))
    log_payload = "\n\n".join(log_sections).rstrip() + "\n"
    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text(log_payload)
        logger.info(f"Metrics summary logged to {args.log_file}")

    if args.save_metrics:
        payload = [_metric_to_dict(m) for m in metrics_list]
        hit_mass_summary = {
            str(r): {"mean": mean_hit, "median": med_hit}
            for r, (mean_hit, med_hit) in hit_stats.items()
        }
        mass_radius_summary = {
            str(p): {"mean": mean_r, "median": med_r}
            for p, (mean_r, med_r) in mass_radius_stats.items()
        }
        args.save_metrics.write_text(json.dumps({
            "metrics": payload,
            "aggregate": {
                "hit_masses": hit_mass_summary,
                "mass_radii": mass_radius_summary,
                "topk_min_dist": {"mean": mean_topk, "median": med_topk},
                "distance_error": {"mean": mean_err, "median": med_err},
                "angular_error_deg": (None if mean_ang is None
                                      else {"mean": mean_ang, "median": med_ang}),
                "iou_error": None if mean_iou_err is None else {"mean": mean_iou_err, "median": med_iou_err},
                "hit_radii": args.hit_radii,
                "mass_percentiles": args.mass_percentiles,
                "top_k_min_dist": args.top_k_min_dist,
                "top_k": args.top_k,
                "grid_step": args.grid_step,
                "matching_weights": {
                    "label_weight": 1.0 - args.relation_weight - args.attribute_weight,
                    "attribute_weight": args.attribute_weight,
                    "relation_weight": args.relation_weight,
                    "neighbor_weight": args.neighbor_weight,
                },
            },
        }, indent=2))
        logger.info(f"Metrics saved to {args.save_metrics}")


GUI_INITIALISED = False

if __name__ == "__main__":
    main()
