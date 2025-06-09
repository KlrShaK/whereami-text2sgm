#!/bin/bash
# SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/graph_models/models/"
# PROJECT_DIR="/home/klrshak/work/VisionLang/whereami-text2sgm/playground/"


# export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
        --mode disabled \
        --epoch 30 \
        --N 1 \
        --lr 0.0001 \
        --weight_decay 0.00005 \
        --batch_size 32 \
        --contrastive_loss True \
        --valid_top_k 1 2 3 5 \
        --use_attributes True \
        --training_with_cross_val True \
        --folds 10 \
        --skip_k_fold True \
        --entire_training_set \
        --eval_iters 10 \
        --eval_iter_count 500 \
        --subgraph_ablation \
        --scanscribe_auto_gen \
        --eval_only_c \
        --model_name 04-05-25_model_NO_subg_100_epochs_30_eval_iters_cut_cross