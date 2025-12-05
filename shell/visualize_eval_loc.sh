#!/bin/bash
# Helper script to run visualize_eval_loc.py with sensible defaults.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"

# 3RScan mesh root (scene folders with meshes + instance labels)
SCENE_ROOT="/home/klrshak/work/VisionLang/3RScan/data/3RScan"

# Caption JSON root (frame-*.json files with ground-truth poses & visible objects)
QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_processed"

# Processed graphs directory (contains processed_data/3dssg/*.pt)
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
SCENE_IDS=()

# Additional CLI options (uncomment / edit as needed)
EXTRA_ARGS=(
  --show_heatmap
  --show_arrows
  --show_3d
  --save_metrics "./eval/eval_metrics.json"
  --frame_policy max_visible
  --top_k 8
  --grid_step 0.25
  --hit_radius 0.5
  --log_file "./eval/eval_loc_summary.log"
  --prediction_strategy "weighted"
)

cd "$PROJECT_DIR" || exit 1

CMD=(
  python visualize_eval_loc_mk2.py
  --root "$SCENE_ROOT"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
)

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
