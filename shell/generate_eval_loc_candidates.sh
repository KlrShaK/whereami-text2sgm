#!/bin/bash
set -euo pipefail

# Candidate pose export using structured frame JSONs.
PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"
# 3RScan example:
# SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan" # Complete dataset
# QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed" # Complete dataset
# GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data" # Complete dataset
# DATASET="3rscan" # Complete dataset

# ScanNet example:
# DATASET="scannet"
# SCENE_ROOT="/media/klrshak/Backup/Datasets/Scannet_300_Scenes"
# QUERY_ROOT="/media/klrshak/Backup/Datasets/Scannet_300_Scenes"
# GRAPHS_DIR="/media/klrshak/Backup/Datasets/scannet_scenes_100/processed_data/generated"
# DATASET="scannet"

# Human annotations dataset: subset of ScanNet with human-authored descriptions.
DATASET="human" # HUMAN ANNOTATIONS
SCENE_ROOT="/media/klrshak/Backup/Datasets/human_scenes" # HUMAN ANNOTATIONS
QUERY_ROOT="/media/klrshak/Backup/Datasets/human_scenes" # HUMAN ANNOTATIONS
GRAPHS_DIR="/media/klrshak/Backup/Datasets/scannet_scenes_100/processed_data/generated" # HUMAN ANNOTATIONS


# Dataset layout for Python loaders: human uses ScanNet layout.
DATASET_LAYOUT="$DATASET"
if [ "$DATASET" = "human" ]; then
  DATASET_LAYOUT="scannet"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_JSON="./eval/eval_pose_candidates_100_seed42_${DATASET}.json"

# FOV defaults per dataset. Human annotations inherit ScanNet camera defaults.
case "$DATASET" in
  scannet|human)
    H_FOV_DEG=58.30   # ScanNet / human
    V_FOV_DEG=45.33   # ScanNet / human
    ;;
  3rscan)
    H_FOV_DEG=39.31   # 3RScan
    V_FOV_DEG=64.76   # 3RScan
    ;;
  *)
    echo "ERROR: Unknown DATASET '$DATASET' (expected 3rscan, scannet, or human)." >&2
    exit 1
    ;;
esac

cd "$PROJECT_DIR" || { echo "Bad PROJECT_DIR: $PROJECT_DIR"; exit 1; }

CMD=(
  "$PYTHON_BIN" generate_eval_loc_candidates.py
  --root "$SCENE_ROOT"
  --dataset "$DATASET_LAYOUT"
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
