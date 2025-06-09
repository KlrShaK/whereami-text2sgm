#!/bin/bash
# SCRIPTPATH=$(dirname $(readlink -f "$0"))
# PROJECT_DIR="/cluster/project/cvg/jiaqchen/h_coarse_loc/playground/graph_models/models/"
PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models/"


# export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

python visualization_graph-object.py \
  --root /home/klrshak/work/VisionLang/3RScan/data/3RScan \
  --graphs /home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/processed_data \
  --ckpt ../model_checkpoints/graph2graph/model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint.pt
  #   --query-key 5341b7bf-8a66-2cdd-8794-026113b7c312 \

