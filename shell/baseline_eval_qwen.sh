#!/bin/bash
# Run Qwen VLM baseline evaluator and store baseline-prefixed outputs in eval/.

PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models"
SCENE_ROOT="/media/klrshak/Backup/Datasets/3RScan_processed"

# Optional: restrict to subset scene IDs (space separated). Leave empty for all.
SCENE_IDS=()

EXTRA_ARGS=(
  --save_metrics "./eval/baseline_eval_metric_qwen.json"
  --log_file "./eval/baseline_eval_metric_qwen.log"
  # --visualize
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
