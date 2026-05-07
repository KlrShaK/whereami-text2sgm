#!/bin/bash
# Evaluate localisation using GPT-parsed cached description graphs (mk5).

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# 3RScan mesh root (scene folders with meshes + instance labels)
# SCENE_ROOT="/home/klrshak/work/VisionLang/3RScan/data/3RScan"
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan"

# Caption JSON root (contains *_parsed.json files after preprocessing)
# QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_processed"
QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Processed graphs directory (contains processed_data/3dssg/*.pt)
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

# Dataset layout for loaders: 3rscan | scannet
DATASET="3rscan"

# # ScanNet example:
# DATASET="scannet"
# SCENE_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"
# QUERY_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"
# GRAPHS_DIR="/media/klrshak/Backup/Datasets/scannet_scenes_100/processed_data/generated"


# Optional: restrict to a subset of scene IDs loaded from file (one per line).
# Override with: SCENE_IDS_FILE=/path/to/scene_ids.txt ./shell/visualize_eval_loc_mk5.sh
SCENE_IDS=()

# RUNNING ON A SUBSET OF SCENES: (comment out if running on all scenes)
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
  --show_heatmap
  --show_arrows
  --show_3d
  --save_candidates "./eval/eval_candidates_mk5_${PREDICTION_STRATEGY}_${DATASET}_SCRATCH.json"
  --save_metrics "./eval/eval_metrics_mk5_${PREDICTION_STRATEGY}_${DATASET}_SCRATCH.json"
  --log_file "./eval/eval_metrics_mk5_${PREDICTION_STRATEGY}_${DATASET}_SCRATCH.log"
  --frame_policy max_visible  # Options: "first", "index", "random", "max_visible", "max_pixels", "all"
  --query_embedding_mode doc
  --homogenize_label_embeddings
  --dynamic_top_k
  --ensure_query_coverage
  --centroid_similarity_threshold 0.7
  --score_threshold 0.1
  --distance_bonus_weight 0.5
  --distance_bonus_decay 2.0
  --grid_step 0.25
  --prediction_strategy "$PREDICTION_STRATEGY"
  --log_level INFO
)

cd "$PROJECT_DIR" || exit 1

CMD=(
  python visualize_eval_loc_mk5.py
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
