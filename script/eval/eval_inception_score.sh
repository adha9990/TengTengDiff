#!/bin/bash

# bash script/eval/eval_inception_score.sh generate_data/hazelnut/full/image

# Inception Score Evaluation Script

IMAGE_DIR="${1:-generate_data/hazelnut/full/image}"
BATCH_SIZE="${2:-32}"
SPLITS="${3:-10}"

echo "=================================================="
echo "Inception Score Evaluation"
echo "=================================================="
echo "Image Directory: $IMAGE_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Splits: $SPLITS"
echo "=================================================="

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory does not exist: $IMAGE_DIR"
    echo "Usage: bash $0 [image_dir] [batch_size] [splits]"
    echo "Example: bash $0 generate_data/hazelnut/full/image 32 10"
    exit 1
fi

.env/bin/python eval/compute_is.py \
    --image_dir "$IMAGE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --splits "$SPLITS"

echo ""
echo "Evaluation completed!"