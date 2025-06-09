#!/bin/bash
# SCRIPTPATH=$(dirname $(readlink -f "$0"))
# PROJECT_DIR="/cluster/project/cvg/jiaqchen/h_coarse_loc/playground/graph_models/models/"
PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models/"


# export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

python3 visualization_minimal.py \
    --scan-id 095821f7-e2c2-2de1-9568-b9ce59920e29 \
    --root /home/klrshak/work/VisionLang/3RScan/data/3RScan/ \
    --graphs /home/klrshak/Downloads/3RScan-Test/graphs/ \
    --ckpt  '/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/model_checkpoints/graph2graph/model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint.pt'