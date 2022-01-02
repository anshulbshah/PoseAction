#!/bin/bash 
python train.py --batch_size 16 \
      --channels 3 --compressed_dim 128 \
      --contrastive_loss_type all --dataset jhmdb \
      --dropout False \
      --contrastive_linear 50 --contrastive_start 50 --learning_rate 1E-4 \
      --loss_weights_contrastive 0.05 --lr_patience 5 \
      --max_motion 7 --max_motion_groupwise 4 \
      --model JMRN \
      --paa True --paa_type global_and_groupwise --n_epochs 100 \
      --n_workers 8 \
      --pose_type openpose_coco_v2 \
      --save_every 5 --temperature 0.3 --train_split 1 \
      --use_background False 