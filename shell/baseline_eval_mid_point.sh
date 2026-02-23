#!/bin/bash
# Run midpoint baseline evaluator and store baseline-prefixed outputs in eval/.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"

# 3RScan mesh root (scene folders with meshes + instance labels)
# SCENE_ROOT="/home/klrshak/work/VisionLang/3RScan/data/3RScan"
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Caption JSON root (frame-*.json files with ground-truth poses & visible objects)
# QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_processed"
# QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_2_2_26_scanscribe"
QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Processed graphs directory (contains processed_data/3dssg/*.pt)
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
# SCENE_IDS=(41385867-a238-2435-8152-dc84ef14eae1)
SCENE_IDS=()

EXTRA_ARGS=(
  --frame_policy all
  # Use --frame_policy all to evaluate every frame JSON in each scene.
  --seed 0
  --random_pitch_deg 30.0
  --save_metrics "./eval/baseline_eval_metrics_mid_point.json"
  --log_file "./eval/baseline_eval_loc_summary_mid_point.log"
)

cd "$PROJECT_DIR" || exit 1

CMD=(
  python3 eval_mid_point_baseline.py
  --root "$SCENE_ROOT"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
)

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
