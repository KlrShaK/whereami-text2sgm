#!/bin/bash
# Evaluate localisation using GPT-parsed cached description graphs (mk5).

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"

# 3RScan mesh root (scene folders with meshes + instance labels)
# SCENE_ROOT="/home/klrshak/work/VisionLang/3RScan/data/3RScan"
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan"

# Caption JSON root (contains *_parsed.json files after preprocessing)
# QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_processed"
QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Processed graphs directory (contains processed_data/3dssg/*.pt)
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
SCENE_IDS=(0ad2d3a5-79e2-2212-9a9e-2502a05fa678 8eabc42c-5af7-2f32-87c4-bf646779aa62 283ccfed-107c-24d5-8b72-5f6004ef4f94 422885ad-192d-25fc-8631-c3a978a9d3d4)
# SCENE_IDS=()

# Additional CLI options (uncomment / edit as needed)
EXTRA_ARGS=(
  # --show_heatmap
  # --show_arrows
  # --show_3d
  --save_metrics "./eval/eval_metrics_mk5.json"
  --log_file "./eval/eval_loc_summary_mk5.log"
  --frame_policy max_visible
  --query_embedding_mode doc
  --homogenize_label_embeddings
  --dynamic_top_k
  --ensure_query_coverage
  --centroid_similarity_threshold 0.7
  --score_threshold 0.1
  --distance_bonus_weight 0.5
  --distance_bonus_decay 2.0
  --grid_step 0.25
  --prediction_strategy "weighted"
)

cd "$PROJECT_DIR" || exit 1

CMD=(
  python visualize_eval_loc_mk5.py
  --root "$SCENE_ROOT"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
)

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
