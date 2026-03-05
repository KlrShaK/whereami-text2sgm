#!/bin/bash
set -euo pipefail

# Generate top-down renders for all scenes.
# Output layout:
#   <DATA_ROOT>/<scene-id>/topdown.png

PROJECT_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm"

DATASET="3rscan"
DATA_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# # ScanNet example:
DATASET="scannet"
DATA_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"

cd "$PROJECT_ROOT" || exit 1

python3 playground/testing/topdown_3rscan.py \
  --root "$DATA_ROOT" \
  --dataset "$DATASET" \
  --output "$DATA_ROOT" \
  --output-name "topdown.png" \
  --all-scans
  # --visualize \
  # --scan-id scene0000_00

