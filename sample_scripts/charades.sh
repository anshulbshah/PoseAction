#!/bin/bash

python train.py --batch_size 128 \
                --compressed_dim 32 \
                --contrastive_loss_type all_charades_atleast_one --dataset charades \
                --dropout True \
                --contrastive_linear 100 --contrastive_start 0 --learning_rate 0.0005 \
                --loss_weights_contrastive 0.05 --lr_patience 2 \
                --max_motion 9 --max_motion_groupwise 5 \
                --paa True --paa_type global_and_groupwise --n_epochs 75 \
                --n_workers 2 \
                --pose_type openpose_coco \
                --temperature 0.1 --train_split 1 \
                --use_background False --use_flip True 