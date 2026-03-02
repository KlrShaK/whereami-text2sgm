#!/bin/bash
# Evaluate localisation using CLIP embeddings + relationship-aware matching (mk6).

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"

# 3RScan mesh root (scene folders with meshes + instance labels)
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan"

# Caption JSON root (contains *_parsed.json files after preprocessing)
QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Processed graphs directory (contains processed_data/3dssg/clip_full_3dssg_graphs.pt)
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
SCENE_IDS=()

# Additional CLI options (uncomment / edit as needed)
EXTRA_ARGS=(
  # --show_heatmap
  # --show_arrows
  # --show_3d
  --save_metrics "./eval/eval_metrics_mk6.json"
  --log_file "./eval/eval_loc_summary_mk6.log"
  --frame_policy max_visible
  --no_homogenize_label_embeddings
  --dynamic_top_k
  --ensure_query_coverage
  --centroid_similarity_threshold 0.7
  --score_threshold 0.05
  --distance_bonus_weight 0.5
  --distance_bonus_decay 2.0
  --grid_step 0.25
  --prediction_strategy "weighted"
  # Relationship-aware matching weights
  --relation_weight 0.3
  --attribute_weight 0.15
  --neighbor_weight 0.5
  # Debug: uncomment to enable per-node match breakdown
  # --debug_match_labels
  # --debug_match_all_scores
  # --debug_match_csv_dir "./eval/match_csvs_mk6"
)

cd "$PROJECT_DIR" || exit 1

CMD=(
  python visualize_eval_loc_mk6.py
  --root "$SCENE_ROOT"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
)

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
