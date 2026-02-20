#!/usr/bin/env python3
"""
preprocess_descriptions.py
==========================
Batch-parse frame description texts via GPT-4o-mini and cache the resulting
scene graphs (with word2vec embeddings) as ``frame-XXXXXX_parsed.json`` files
alongside the original ``frame-XXXXXX.json`` files.

Usage:
    python preprocess_descriptions.py \
        --data_root /path/to/3RScan_processed \
        --api_key_file /path/to/openai_key.txt \
        --max_frames 5 --dry_run

Or set OPENAI_API_KEY and omit --api_key_file:
    export OPENAI_API_KEY=sk-...
    python preprocess_descriptions.py --data_root /path/to/3RScan_processed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import openai

# --------------------------------------------------------------------------- #
# Repository imports                                                          #
# --------------------------------------------------------------------------- #

sys.path.append("../data_processing")
sys.path.append("../../../")

from create_text_embeddings import create_embedding_nlp  # noqa: E402
from graph_loader_utils import get_word2vec  # noqa: E402

# Reuse the GPT parser from single_inference.py
from single_inference import parse_text_to_json  # noqa: E402


# --------------------------------------------------------------------------- #
# word2vec embedding (same caching pattern as visualize_eval_loc_mk4.py)      #
# --------------------------------------------------------------------------- #

_EMBED_CACHE: Dict[str, np.ndarray] = {}
_EMBED_CACHE_TOKEN: Dict[str, np.ndarray] = {}
_W2V_HASH: Dict[str, np.ndarray] = {}


def _embed_word2vec(text: str, mode: str = "token") -> List[float]:
    text = str(text)
    key = text.strip().lower()
    if mode == "doc":
        cached = _EMBED_CACHE.get(key)
        if cached is None:
            vec = np.asarray(create_embedding_nlp(text), dtype=np.float32)
            cached = vec
            _EMBED_CACHE[key] = cached
        return cached.tolist()

    cached = _EMBED_CACHE_TOKEN.get(key)
    if cached is None:
        w2v = get_word2vec(text, _W2V_HASH)
        vec = w2v[0] if isinstance(w2v, tuple) else w2v
        cached = np.asarray(vec, dtype=np.float32)
        _EMBED_CACHE_TOKEN[key] = cached
    return cached.tolist()


# --------------------------------------------------------------------------- #
# Discover frame JSONs                                                        #
# --------------------------------------------------------------------------- #

def discover_frame_jsons(data_root: Path,
                         scene_ids: Optional[List[str]] = None) -> List[Path]:
    """Return all frame-*.json paths (excluding *_parsed.json) under data_root."""
    if scene_ids:
        dirs = []
        for sid in scene_ids:
            d = data_root / sid / "output" / "descriptions"
            if d.exists():
                dirs.append(d)
    else:
        dirs = sorted(data_root.glob("*/output/descriptions"))

    paths: List[Path] = []
    for d in dirs:
        for p in sorted(d.glob("frame-*.json")):
            if p.stem.endswith("_parsed"):
                continue
            paths.append(p)
    return paths


def parsed_path_for(frame_path: Path) -> Path:
    """Return the *_parsed.json path corresponding to a frame JSON."""
    return frame_path.with_name(frame_path.stem + "_parsed.json")


# --------------------------------------------------------------------------- #
# Process a single frame                                                      #
# --------------------------------------------------------------------------- #

def process_frame(frame_path: Path,
                  embedding_mode: str,
                  dry_run: bool = False) -> Optional[dict]:
    """Parse description via GPT, embed with word2vec, return parsed dict."""
    data = json.loads(frame_path.read_text())
    description = data.get("description", "")
    if not description.strip():
        print(f"  [SKIP] No description in {frame_path.name}")
        return None

    scene_index = data.get("scene_index", "")
    image_index = data.get("image_index", frame_path.stem)

    if dry_run:
        print(f"  [DRY RUN] Would parse: {frame_path.name} "
              f"(scene={scene_index}, desc length={len(description)})")
        return None

    # Call GPT to parse
    parsed = parse_text_to_json(description)

    # Embed nodes
    for node in parsed.get("nodes", []):
        node["label_word2vec"] = _embed_word2vec(node["label"], mode=embedding_mode)
        node["attributes_word2vec"] = {
            "all": [_embed_word2vec(a, mode=embedding_mode)
                    for a in node.get("attributes", [])]
        }

    # Embed edges
    for edge in parsed.get("edges", []):
        edge["relation_word2vec"] = _embed_word2vec(
            edge["relationship"], mode=embedding_mode
        )

    result = {
        "source_frame": image_index,
        "scene_index": scene_index,
        "description": description,
        "parsed_graph": parsed,
    }
    return result


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess frame descriptions via GPT and cache parsed scene graphs."
    )
    p.add_argument("--data_root", required=True, type=Path,
                   help="Root of 3RScan_processed (contains <scene_id>/output/descriptions/).")
    p.add_argument("--api_key_file", type=Path,
                   help="Optional path to file with line 'OPENAI_API_KEY=sk-...' or just the key. "
                        "If omitted, OPENAI_API_KEY env var is used.")
    p.add_argument("--embedding_mode", choices=["token", "doc"], default="token",
                   help="word2vec embedding mode: 'token' (first-token) or 'doc' (spaCy doc.vector).")
    p.add_argument("--scene_ids", nargs="+",
                   help="Optional list of scene IDs to restrict processing.")
    p.add_argument("--dry_run", action="store_true",
                   help="Preview which frames would be processed without calling GPT.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-process frames even if *_parsed.json already exists.")
    p.add_argument("--max_frames", type=int,
                   help="Limit the number of frames to process (for testing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load API key from optional file, otherwise OPENAI_API_KEY.
    key = ""
    if args.api_key_file is not None:
        with open(args.api_key_file, "r") as f:
            line = f.read().strip()
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1]
            else:
                key = line
    else:
        key = os.getenv("OPENAI_API_KEY", "").strip()

    if not key:
        raise ValueError(
            "No OpenAI API key found. Set OPENAI_API_KEY or pass --api_key_file."
        )

    openai.api_key = key

    # Discover frames
    all_frames = discover_frame_jsons(args.data_root, args.scene_ids)
    print(f"Found {len(all_frames)} frame JSON(s) under {args.data_root}")

    # Filter already-processed (unless --overwrite)
    if not args.overwrite:
        todo = [p for p in all_frames if not parsed_path_for(p).exists()]
        skipped = len(all_frames) - len(todo)
        if skipped:
            print(f"Skipping {skipped} already-parsed frame(s) (use --overwrite to redo).")
    else:
        todo = all_frames

    if args.max_frames is not None:
        todo = todo[: args.max_frames]

    print(f"Processing {len(todo)} frame(s)...\n")

    success = 0
    errors = 0
    for idx, frame_path in enumerate(todo, start=1):
        scene_id = frame_path.parts[-4]  # .../scene_id/output/descriptions/frame-*.json
        print(f"[{idx:04d}/{len(todo):04d}] {scene_id}/{frame_path.name}")

        try:
            result = process_frame(frame_path, args.embedding_mode, dry_run=args.dry_run)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            errors += 1
            continue

        if result is None:
            continue

        out_path = parsed_path_for(frame_path)
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  -> saved {out_path.name} "
              f"({len(result['parsed_graph']['nodes'])} nodes, "
              f"{len(result['parsed_graph']['edges'])} edges)")
        success += 1

        # Small delay to avoid rate limits
        time.sleep(0.1)

    print(f"\nDone. Success: {success} | Errors: {errors} | "
          f"Dry-run skipped: {len(todo) - success - errors}")


if __name__ == "__main__":
    main()
