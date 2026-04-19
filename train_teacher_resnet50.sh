#!/bin/bash
# ResNet-50 Teacher Training on CIFAR100
# This script trains ResNet-50 as a teacher model on CIFAR100

# Activate conda environment
conda activate pydev

# Single GPU training with torchrun
PYTHONPATH=. torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  tools/train.py \
  -c configs/strategies/CIFAR/cifar100.yaml \
  --model nas_model \
  --model-config configs/models/ResNet/resnet-50-cifar100.yaml \
  --experiment teacher_resnet50_cifar100

# Alternative for multi-GPU (if available):
# sh tools/dist_train.sh 1 configs/strategies/CIFAR/cifar100.yaml nas_model \
#   --model-config configs/models/ResNet/resnet-50-cifar100.yaml \
#   --experiment teacher_resnet50_cifar100
