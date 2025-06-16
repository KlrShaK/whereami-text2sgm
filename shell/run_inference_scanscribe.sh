#!/bin/bash
# ------------------------------------------------------------------
# Path configuration – adapt only these two lines if needed
PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models/"
DATA_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"
# ------------------------------------------------------------------

cd "$PROJECT_DIR" || { echo "Bad PROJECT_DIR"; exit 1; }

python inference.py \
  --graphs  "$DATA_DIR" \
  --ckpt    ../model_checkpoints/graph2graph/model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint.pt \
  --top_k   5 \
  --device  cuda \
  --jsonl_out  scanscribe_top5.jsonl
