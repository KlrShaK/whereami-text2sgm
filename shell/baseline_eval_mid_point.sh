#!/bin/bash
# Run midpoint baseline evaluator and store baseline-prefixed outputs in eval/.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# 3RScan mesh root (scene folders with meshes + instance labels)
# SCENE_ROOT="/home/klrshak/work/VisionLang/3RScan/data/3RScan"
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Caption JSON root (frame-*.json files with ground-truth poses & visible objects)
# QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_processed"
# QUERY_ROOT="/home/klrshak/work/VisionLang/whereami-text2sgm/datasets/3RScan_2_2_26_scanscribe"
QUERY_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Optional processed graphs directory (contains processed_data/3dssg/*.pt).
# Leave empty for datasets without processed graph exports (e.g., ScanNet).
GRAPHS_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

# Dataset layout for loaders: 3rscan | scannet
DATASET="3rscan"

# # ScanNet example:
# DATASET="scannet"
# SCENE_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"
# QUERY_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"
# GRAPHS_DIR=""


# Optional: restrict to a subset of scene IDs (space separated). Leave empty for all.
# SCENE_IDS=(41385867-a238-2435-8152-dc84ef14eae1)
# SCENE_IDS=(scene0003_02 scene0051_01)
SCENE_IDS=()

# # RUNNING ON A SUBSET OF SCENES: (COMMENT out if running on all scenes)
# SCENE_IDS_FILE="${SCENE_IDS_FILE:-$REPO_ROOT/playground/testing/subset_100_scene_ids.txt}"
# if [ -f "$SCENE_IDS_FILE" ]; then
#   mapfile -t SCENE_IDS < <(grep -vE '^[[:space:]]*(#|$)' "$SCENE_IDS_FILE")
#   echo "[INFO] Loaded ${#SCENE_IDS[@]} scene IDs from $SCENE_IDS_FILE"
# else
#   echo "[WARN] SCENE_IDS_FILE not found: $SCENE_IDS_FILE (running on all scenes)"
# fi


# FOV defaults per dataset
if [ "$DATASET" = "scannet" ]; then
  H_FOV_DEG=58.30   # ScanNet
  V_FOV_DEG=45.33   # ScanNet
else
  H_FOV_DEG=39.31   # 3RScan
  V_FOV_DEG=64.76   # 3RScan
fi

EXTRA_ARGS=(
  # --show_3d
  --frame_policy all
  # Use --frame_policy all to evaluate every frame JSON in each scene.
  --seed 42
  --log_level INFO
  --random_pitch_deg 30.0
  --save_metrics "./eval/baseline_eval_metrics_mid_point_${DATASET}.json"
  --log_file "./eval/baseline_eval_metrics_mid_point_${DATASET}.log"
)

cd "$PROJECT_DIR" || exit 1

CMD=(
  python3 eval_mid_point_baseline.py
  --root "$SCENE_ROOT"
  --dataset "$DATASET"
  --query_root "$QUERY_ROOT"
  --h_fov_deg "$H_FOV_DEG"
  --v_fov_deg "$V_FOV_DEG"
)

if [ -n "$GRAPHS_DIR" ]; then
  CMD+=(--graphs "$GRAPHS_DIR")
fi

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
