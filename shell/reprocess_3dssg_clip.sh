#!/bin/bash
# Reprocess 3DSSG graphs with all-CLIP embeddings (label_clip, relation_clip, attributes_clip).
# Output: clip_full_3dssg_graphs.pt

SCRIPT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/data_processing"
DATA_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data"

INPUT_PT="${DATA_DIR}/3dssg/3dssg_graphs_processed_edgelists_relationembed.pt"
OUTPUT_PT="${DATA_DIR}/3dssg/clip_full_3dssg_graphs.pt"

cd "$SCRIPT_DIR" || exit 1

python reprocess_3dssg_clip.py \
  --input_pt "$INPUT_PT" \
  --output_pt "$OUTPUT_PT" \
  --batch_size 64
