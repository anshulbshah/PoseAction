#!/bin/bash 
python train.py --batch_size 16 \
      --channels 3 --compressed_dim 128 \
      --contrastive_loss_type all --dataset jhmdb \
      --dropout False --eca_module sigmoid_conv_relu2 \
      --l1_linear 50 --l1_start 50 --learning_rate 1E-4 \
      --loss_weights_contrastive 0.05 --lr_patience 5 \
      --max_motion 7 --max_motion_groupwise 4 \
      --model JMRN \
      --paa True --paa_type global_and_groupwise --n_epochs 100 \
      --n_workers 8 --normalize True --normalize_type area --optimizer adam \
      --pose_type openpose_coco_v2 --reduced_dim 512 --return_augmented_view True \
      --save_every 5 --scheduler on_plateau --temperature 0.3 --train_split 1 \
      --trainer_type ch_wt_contrastive --use_background False --gumbel_val False \
      --gumbel_temperature 0.66