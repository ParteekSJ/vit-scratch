#!/bin/bash
#
# ViT-Base Training Script
#

# Set experiment name and configuration
EXP_NAME="$(date +%Y-%m-%d-%H-%M-%S)"
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"

# Create directories
mkdir -p ${CHECKPOINT_DIR}

# Start training
echo "Starting training for CIFAR10 dataset."
python main.py \
  --in_channels 3 \
  --img_size 32 \
  --patch_size 4 \
  --embedding_dim 768 \
  --depth 12 \
  --num_heads 12 \
  --mlp_ratio 4 \
  --num_classes 10 \
  --dropout_rate 0.1 \
  --attn_dropout_rate 0.0 \
  --dataset cifar10 \
  --data_dir data \
  --batch_size 4 \
  --epochs 100 \
  --learning_rate 3e-4 \
  --weight_decay 0.05 \
  --warmup_steps 500 \
  --seed 42 \
  --save_dir ${CHECKPOINT_DIR} \
  --save_freq 5 \
  --print_freq 20 

echo "Training completed. Results saved in ${CHECKPOINT_DIR}"