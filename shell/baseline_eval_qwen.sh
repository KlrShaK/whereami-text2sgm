#!/bin/bash
# Run Qwen VLM baseline evaluator and store baseline-prefixed outputs in eval/.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"

# 3RScan root (scene folders with topdown.png, topdown_camera.npz, output/descriptions)
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Dataset tag used in output filenames.
DATASET="3rscan"

# Optional: restrict to subset scene IDs (space separated). Leave empty for all.
SCENE_IDS=(41385867-a238-2435-8152-dc84ef14eae1)
# SCENE_IDS=()

EXTRA_ARGS=(
  --seed 0
  --h_fov_deg 39.31
  --v_fov_deg 64.76
  --save_metrics "./eval/baseline_eval_metrics_qwen_${DATASET}.json"
  --log_file "./eval/baseline_eval_metrics_qwen_${DATASET}.log"
  # --resume
  # --max_frames_per_scene 10
  --visualize
)

cd "$PROJECT_DIR" || exit 1

CMD=(
  python3 vlm_baseline.py
  --root "$SCENE_ROOT"
)

if [ ${#SCENE_IDS[@]} -gt 0 ]; then
  CMD+=(--scene_ids "${SCENE_IDS[@]}")
fi

CMD+=("${EXTRA_ARGS[@]}")

"${CMD[@]}"
