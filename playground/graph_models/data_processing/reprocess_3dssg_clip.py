#!/usr/bin/env python3
"""
reprocess_3dssg_clip.py
-----------------------
Load an existing 3DSSG processed .pt file and re-embed all labels, attributes,
and edge relations using CLIP (512-dim) under new key names:

  label_clip, attributes_clip, relation_clip

Existing word2vec keys are preserved.  The output is saved as a new .pt file
(default: clip_full_3dssg_graphs.pt).

Usage:
    python reprocess_3dssg_clip.py \
        --input_pt  ../processed_data/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt \
        --output_pt ../processed_data/3dssg/clip_full_3dssg_graphs.pt \
        --batch_size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from create_text_embeddings import (  # noqa: E402
    create_embedding_clip,
    create_embeddings_clip_batch,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _to_list(tensor_or_list):
    """Convert a tensor to a Python list (nested) for serialisation."""
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.tolist()
    if hasattr(tensor_or_list, "tolist") and not isinstance(tensor_or_list, str):
        return tensor_or_list.tolist()
    return tensor_or_list


def _batch_embed_texts(texts: list[str], batch_size: int, cache: dict[str, list[float]]) -> list[list[float]]:
    """Embed a list of texts with CLIP, using a cache to avoid duplicates."""
    results: list[list[float] | None] = [None] * len(texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, t in enumerate(texts):
        key = t.strip().lower()
        if key in cache:
            results[i] = cache[key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(t)

    # Batch-embed uncached texts
    for start in range(0, len(uncached_texts), batch_size):
        batch = uncached_texts[start : start + batch_size]
        embeddings = create_embeddings_clip_batch(batch)  # (B, 512)
        for j, emb in enumerate(embeddings):
            idx = uncached_indices[start + j]
            emb_list = _to_list(emb)
            if isinstance(emb_list, str):
                # Some environments may return unexpected iterable payloads from
                # the batch helper; fall back to single-text embedding to keep
                # reprocessing robust.
                emb_list = create_embedding_clip(batch[j]).tolist()
            results[idx] = emb_list
            key = uncached_texts[start + j].strip().lower()
            cache[key] = emb_list

    return results  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Main reprocessing                                                            #
# --------------------------------------------------------------------------- #

def reprocess(input_pt: Path, output_pt: Path, batch_size: int) -> None:
    print(f"Loading {input_pt} ...")
    # PyTorch 2.6 defaults torch.load(..., weights_only=True), which rejects
    # these trusted local graph pickles (they are not plain tensor state dicts).
    all_scenes = torch.load(input_pt, map_location="cpu", weights_only=False)
    print(f"  {len(all_scenes)} scenes loaded.")

    cache: dict[str, list[float]] = {}

    # ---- Pass 1: Node labels & attributes --------------------------------- #
    print("Pass 1/2: Embedding node labels and attributes with CLIP ...")
    for scene_id in tqdm(all_scenes, desc="Nodes"):
        objects = all_scenes[scene_id].get("objects", {})
        # Collect all label texts for batch embedding
        obj_ids = list(objects.keys())
        label_texts = [objects[oid]["label"] for oid in obj_ids]
        label_embs = _batch_embed_texts(label_texts, batch_size, cache)

        for oid, label_emb in zip(obj_ids, label_embs):
            objects[oid]["label_clip"] = label_emb

            # Attributes: dict of {attr_type: [str, ...]}
            raw_attrs = objects[oid].get("attributes", {})
            attributes_clip: dict[str, list[list[float]]] = {}
            for attr_type, attr_vals in raw_attrs.items():
                if attr_vals:
                    attr_embs = _batch_embed_texts(attr_vals, batch_size, cache)
                    attributes_clip[attr_type] = attr_embs
                else:
                    attributes_clip[attr_type] = []
            objects[oid]["attributes_clip"] = attributes_clip

    # ---- Pass 2: Edge relations ------------------------------------------- #
    print("Pass 2/2: Embedding edge relations with CLIP ...")
    for scene_id in tqdm(all_scenes, desc="Edges"):
        edge_lists = all_scenes[scene_id].get("edge_lists", {})
        relations = edge_lists.get("relation", [])
        if not relations:
            edge_lists["relation_clip"] = []
            continue
        rel_embs = _batch_embed_texts(relations, batch_size, cache)
        edge_lists["relation_clip"] = rel_embs

    # ---- Save ------------------------------------------------------------- #
    output_pt.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_pt} ...")
    torch.save(all_scenes, str(output_pt))

    # Quick verification
    sample_sid = next(iter(all_scenes))
    sample_obj = next(iter(all_scenes[sample_sid]["objects"].values()))
    dim = len(sample_obj["label_clip"])
    print(f"  Verification: label_clip dim = {dim}")
    edge_lists = all_scenes[sample_sid].get("edge_lists", {})
    if edge_lists.get("relation_clip"):
        rdim = len(edge_lists["relation_clip"][0])
        print(f"  Verification: relation_clip dim = {rdim}")
    print("Done.")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reprocess 3DSSG graphs with CLIP embeddings (label_clip, relation_clip, attributes_clip)."
    )
    parser.add_argument(
        "--input_pt", type=Path, required=True,
        help="Path to existing 3DSSG .pt file with edge_lists.",
    )
    parser.add_argument(
        "--output_pt", type=Path, required=True,
        help="Output .pt file path.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for CLIP embedding (default: 64).",
    )
    args = parser.parse_args()
    reprocess(args.input_pt, args.output_pt, args.batch_size)


if __name__ == "__main__":
    main()
