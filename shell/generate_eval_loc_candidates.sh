#!/bin/bash
set -euo pipefail

# Candidate pose export using structured frame JSONs.
PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan"
QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"
DATASET="3rscan"

# ScanNet example:
# DATASET="scannet"
# SCENE_ROOT="/media/klrshak/Backup/Datasets/Scannet_300_Scenes"
# QUERY_ROOT="/media/klrshak/Backup/Datasets/Scannet_300_Scenes"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_JSON="./eval/eval_pose_candidates_100_seed42_${DATASET}.json"

# FOV defaults per dataset
if [ "$DATASET" = "scannet" ]; then
  H_FOV_DEG=58.30   # ScanNet
  V_FOV_DEG=45.33   # ScanNet
else
  H_FOV_DEG=39.31   # 3RScan
  V_FOV_DEG=64.76   # 3RScan
fi

cd "$PROJECT_DIR" || { echo "Bad PROJECT_DIR: $PROJECT_DIR"; exit 1; }

CMD=(
  "$PYTHON_BIN" generate_eval_loc_candidates.py
  --root "$SCENE_ROOT"
  --dataset "$DATASET"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
  --frame_policy max_visible
  --max_scenes 300
  --scene_sample_policy random
  --seed 42
  --output_json "$OUTPUT_JSON"
  --h_fov_deg "$H_FOV_DEG"
  --v_fov_deg "$V_FOV_DEG"
)

CMD+=("$@")
"${CMD[@]}"
