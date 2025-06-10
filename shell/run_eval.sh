#!/bin/bash
# SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models/"
# PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/"


# export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# CUDA_VISIBLE_DEVICES=0 python3 eval.py \
#         --mode disabled \
#         --epoch 30 \
#         --N 1 \
#         --lr 0.0001 \
#         --weight_decay 0.00005 \
#         --batch_size 32 \
#         --contrastive_loss True \
#         --valid_top_k 1 2 3 5 \
#         --use_attributes True \
#         --training_with_cross_val True \
#         --folds 10 \
#         --skip_k_fold True \
#         --entire_training_set \
#         --eval_iters 10 \
#         --eval_iter_count 500 \
#         --subgraph_ablation \
#         --scanscribe_auto_gen \
#         --eval_only_c \
#         --model_name model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
    --epoch 100 \
    --N 1 \
    --lr 0.0001 \
    --weight_decay 0.00005 \
    --overlap_thr 0.8 \
    --cos_sim_thr 0.5 \
    --batch_size 16 \
    --training_set_size 3159 \
    --test_set_size 1116 \
    --contrastive_loss True \
    --valid_top_k 1 2 3 5 \
    --use_attributes True \
    --training_with_cross_val True \
    --folds 10 \
    --skip_k_fold True \
    --entire_training_set \
    --eval_iters 100 \
    --subgraph_ablation \
    --heads 2 \
    --model_name model_NO_subg_100_epochs_entire_training_set_epoch_30_checkpoint
