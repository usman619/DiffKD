#!/bin/bash
# Monitor ResNet-50 training on CIFAR100

echo "=== ResNet-50 Teacher Training Monitor ==="
echo "Experiment: teacher_resnet50_cifar100"
echo ""
echo "Training started at: $(date)"
echo ""
echo "Watching log file..."
tail -f experiments/teacher_resnet50_cifar100/log.txt
