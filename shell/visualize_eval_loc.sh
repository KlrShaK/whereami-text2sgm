#!/bin/bash
# Helper script to run visualize_eval_loc.py with sensible defaults.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"

# 3RScan mesh root (scene folders with meshes + instance labels)
# SCENE_ROOT="/home/klrshak/work/VisionLang/3RScan/data/3RScan"
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan"


# Caption JSON root (frame-*.json files with ground-truth poses & visible objects)
QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_processed"
# QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_2_2_26_scanscribe"

# Processed graphs directory (contains processed_data/3dssg/*.pt)
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
# SCENE_IDS=(41385867-a238-2435-8152-dc84ef14eae1 )
# SCENE_IDS=(0cac755a-8d6f-2d13-8fed-b1be02f4ef77 0cac7564-8d6f-2d13-8cb2-8b01c0a1b3d5)
SCENE_IDS=()


# Additional CLI options (uncomment / edit as needed)
EXTRA_ARGS=(
  # --show_heatmap
  # --show_arrows
  # --show_3d
  # --coarse_disable_nms # Disable NMS for coarse localization
  --save_metrics "./eval/eval_metrics.json"
  --log_file "./eval/eval_loc_summary.log"
  --frame_policy max_visible
  --top_k 10
  --grid_step 0.25 #2.0
  # --hit_radius 2.0
  --prediction_strategy "weighted"
)

cd "$PROJECT_DIR" || exit 1

CMD=(
  python visualize_eval_loc.py
  --root "$SCENE_ROOT"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
)

# CMD=(
#   python visualize_eval_loc_mk2.py
#   --root "$SCENE_ROOT"
#   --graphs "$GRAPHS_DIR"
#   --query_root "$QUERY_ROOT"
# )

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
