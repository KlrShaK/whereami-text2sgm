#!/bin/bash
# Visualize IoU with textured mesh from mk5 candidates JSON.
# Supports both ScanNet and 3RScan datasets — toggle DATASET below.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"

# ---- Dataset selection: "scannet" or "3rscan" ----
DATASET="${DATASET:-scannet}"
DATASET="scannet"  # Uncomment to override default

# ---- Per-dataset defaults ----
if [ "$DATASET" = "scannet" ]; then
  SCENE_ROOT="/media/klrshak/Backup/Datasets/scannet_scenes_100/scans"
  CANDIDATES="$PROJECT_DIR/eval/eval_candidates_mk5_weighted_scannet_abu_final.json"
  H_FOV_DEG=58.30
  V_FOV_DEG=45.33
elif [ "$DATASET" = "3rscan" ]; then
  SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan"
  CANDIDATES="$PROJECT_DIR/eval/eval_candidates_mk5_weighted_3rscan_abu_final.json"
  H_FOV_DEG=39.31
  V_FOV_DEG=64.76
else
  echo "[ERROR] Unknown DATASET='$DATASET'. Use 'scannet' or '3rscan'."
  exit 1
fi

# Override scene ID with first positional arg if provided
SCENE_ID="${1:-$SCENE_ID}"

# Frame ID (optional, pass as second arg)
FRAME_ID="${2:-}"

cd "$PROJECT_DIR" || exit 1

CMD=(
  python visualize_iou_textured.py
  --candidates "$CANDIDATES"
  --root "$SCENE_ROOT"
  --dataset "$DATASET"
  --scene_id "$SCENE_ID"
  --h_fov_deg "$H_FOV_DEG"
  --v_fov_deg "$V_FOV_DEG"
  --frustum_scale 0.8
  --tint_strength 0.45
  --desc_wrap_width 100
  --no_remove_roof
)

if [ -n "$FRAME_ID" ]; then
  CMD+=(--frame_id "$FRAME_ID")
fi

"${CMD[@]}"
