#!/bin/bash
# Evaluate localisation using CLIP embeddings + relationship-aware matching (mk6).

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"


# 3RScan mesh root (scene folders with meshes + instance labels)
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan"

# Caption JSON root (contains *_parsed.json files after preprocessing)
QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Processed graphs directory (contains processed_data/3dssg/clip_full_3dssg_graphs.pt)
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

# Dataset layout for loaders: 3rscan | scannet
DATASET="3rscan"

# ScanNet example:
DATASET="scannet"
SCENE_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"
QUERY_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"
GRAPHS_DIR="/media/klrshak/Backup/Datasets/scannet_scenes_100/processed_data/generated"

# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
SCENE_IDS=()

# # RUNNING ON A SUBSET OF SCENES: (comment out if running on all scenes)
# SCENE_IDS_FILE="${SCENE_IDS_FILE:-$REPO_ROOT/playground/testing/subset_100_scene_ids.txt}"
# if [ -f "$SCENE_IDS_FILE" ]; then
#   mapfile -t SCENE_IDS < <(grep -vE '^[[:space:]]*(#|$)' "$SCENE_IDS_FILE")
#   echo "[INFO] Loaded ${#SCENE_IDS[@]} scene IDs from $SCENE_IDS_FILE"
# else
#   echo "[WARN] SCENE_IDS_FILE not found: $SCENE_IDS_FILE (running on all scenes)"
# fi



PREDICTION_STRATEGY="weighted"  # Options: "argmax", "random", "weighted"

# FOV defaults per dataset
if [ "$DATASET" = "scannet" ]; then
  H_FOV_DEG=58.30   # ScanNet
  V_FOV_DEG=45.33   # ScanNet
else
  H_FOV_DEG=39.31   # 3RScan
  V_FOV_DEG=64.76   # 3RScan
fi

# Additional CLI options (uncomment / edit as needed)
EXTRA_ARGS=(
  # --show_heatmap
  # --show_arrows
  # --show_3d
  --save_metrics "./eval/eval_metrics_mk6_${PREDICTION_STRATEGY}_${DATASET}.json"
  --log_file "./eval/eval_loc_summary_mk6_${PREDICTION_STRATEGY}_${DATASET}.log"
  --frame_policy max_visible
  --no_homogenize_label_embeddings
  --dynamic_top_k
  --ensure_query_coverage
  --centroid_similarity_threshold 0.60
  --score_threshold 0.03
  --distance_bonus_weight 0.6
  --distance_bonus_decay 2.5
  --grid_step 0.25
  --score_tau 1.0
  --cluster_bandwidth 0.75
  --prediction_strategy "$PREDICTION_STRATEGY"
  # Relationship-aware matching weights (label_w = 1 - 0.25 - 0.10 = 0.65)
  --relation_weight 0.25
  --attribute_weight 0.10
  --neighbor_weight 0.55
  # Debug: uncomment to enable per-node match breakdown
  # --debug_match_labels
  # --debug_match_all_scores
  # --debug_match_csv_dir "./eval/match_csvs_mk6"
)

cd "$PROJECT_DIR" || exit 1

CMD=(
  python visualize_eval_loc_mk6.py
  --root "$SCENE_ROOT"
  --dataset "$DATASET"
  --graphs "$GRAPHS_DIR"
  --query_root "$QUERY_ROOT"
  --h_fov_deg "$H_FOV_DEG"
  --v_fov_deg "$V_FOV_DEG"
)

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
