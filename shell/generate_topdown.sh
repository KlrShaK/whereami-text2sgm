#!/bin/bash
set -euo pipefail

# Generate top-down renders for all 3RScan_processed scenes.
# Output layout:
#   <DATA_ROOT>/<scene-id>/topdown.png

PROJECT_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm"
DATA_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

cd "$PROJECT_ROOT" || exit 1

python3 playground/testing/topdown_3rscan.py \
  --root "$DATA_ROOT" \
  --all-scans \
  --output "$DATA_ROOT" \
  --output-name "topdown.png"
