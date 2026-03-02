#!/bin/bash
set -euo pipefail

# Candidate pose export using structured frame JSONs.
PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan"
QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_JSON="./eval/eval_pose_candidates_300_seed42.json"

cd "$PROJECT_DIR" || { echo "Bad PROJECT_DIR: $PROJECT_DIR"; exit 1; }

CMD=(
  "$PYTHON_BIN" visualize_eval_loc_candidates.py
  --root "$SCENE_ROOT"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
  --frame_policy max_visible
  --max_scenes 300
  --scene_sample_policy random
  --seed 42
  --output_json "$OUTPUT_JSON"
)

CMD+=("$@")
"${CMD[@]}"
