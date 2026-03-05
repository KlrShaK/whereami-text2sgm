#!/bin/bash
# Helper script to run visualize_eval_loc.py with sensible defaults.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"

# 3RScan mesh root (scene folders with meshes + instance labels)
# SCENE_ROOT="/home/klrshak/work/VisionLang/3RScan/data/3RScan"
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan"


# Caption JSON root (frame-*.json files with ground-truth poses & visible objects)
# QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_processed"
QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_2_2_26_scanscribe"

# Processed graphs directory (contains processed_data/3dssg/*.pt)
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

# Dataset layout for loaders: 3rscan | scannet
DATASET="3rscan"

# FOV defaults per dataset
if [ "$DATASET" = "scannet" ]; then
  H_FOV_DEG=58.30   # ScanNet
  V_FOV_DEG=45.33   # ScanNet
else
  H_FOV_DEG=39.31   # 3RScan
  V_FOV_DEG=64.76   # 3RScan
fi

# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
# SCENE_IDS=(41385867-a238-2435-8152-dc84ef14eae1 )
# SCENE_IDS=(5341b7d3-8a66-2cdd-8633-0a3da632befa 0cac75b1-8d6f-2d13-8c17-9099db8915bc e61b0e04-bada-2f31-82d6-72831a602ba7 1d234004-e280-2b1a-8ec8-560046b9fc96 0cac75dc-8d6f-2d13-8d08-9c497bd6acdc)
SCENE_IDS=()


# Additional CLI options (uncomment / edit as needed)
EXTRA_ARGS=(
  # --show_heatmap
  # --show_arrows
  # --show_3d
  # --coarse_disable_nms # Disable NMS for coarse localization
  --save_metrics "./eval/eval_metrics_mk3.json"
  --log_file "./eval/eval_loc_summary_mk3.log"
  --frame_policy max_visible
  --query_embedding_mode token
  --homogenize_label_embeddings
  # --exact_label_score 1.0
  # --no_homogenize_label_embeddings
  # ---------
  # --scene_use_attributes
  # --debug_match_labels
  # --debug_match_topn 5
  # --debug_match_all_scores
  # --debug_match_csv_dir "./eval/baseline_match_scores"
  --dynamic_top_k
  --ensure_query_coverage
  
  # ----------------
  --score_threshold 0.05
  --distance_bonus_weight 0.5
  --distance_bonus_decay 2.0
  # --top_k 10  # used only when --dynamic_top_k is not set
  --grid_step 0.25 #2.0
  # --hit_radius 2.0
  --prediction_strategy "weighted"
)

cd "$PROJECT_DIR" || exit 1

# CMD=(
#   python visualize_eval_loc.py
#   --root "$SCENE_ROOT"
#   --graphs "$GRAPHS_DIR"
#   --query_root "$QUERY_ROOT"
# )

CMD=(
  python visualize_eval_loc_mk4.py
  --root "$SCENE_ROOT"
  --dataset "$DATASET"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
  --h_fov_deg "$H_FOV_DEG"
  --v_fov_deg "$V_FOV_DEG"
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
