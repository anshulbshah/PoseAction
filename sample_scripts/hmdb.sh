#!/bin/bash
python train.py --batch_size 128 \
                --channels 3 --compressed_dim 64 \
                --contrastive_loss_type all --dataset hmdb \
                --dropout False \
                --contrastive_linear 100 --contrastive_start 0 --learning_rate 0.0001 \
                --loss_weights_contrastive 0.5 --lr_patience 3 \
                --max_motion 5 --max_motion_groupwise 0 \
                --model JMRN \
                --paa True --paa_type global_and_groupwise --n_epochs 150 \
                --n_workers 8 \
                --pose_type openpose_coco_v2 \
                --temperature 0.1 --train_split 1 \
                --use_background False --use_flip True 