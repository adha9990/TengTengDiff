#!/bin/bash

# bash script/eval/eval_ic_lpips.sh generate_data/hazelnut/full/image

# IC-LPIPS (Intra-cluster pairwise LPIPS) Evaluation Script

IMAGE_DIR="${1:-generate_data/hazelnut/full/image}"
LPIPS_NET="${2:-alex}"
MAX_IMAGES="${3:-}"
BATCH_SIZE="${4:-32}"

echo "=================================================="
echo "IC-LPIPS Evaluation"
echo "=================================================="
echo "Image Directory: $IMAGE_DIR"
echo "LPIPS Network: $LPIPS_NET"
if [ -n "$MAX_IMAGES" ]; then
    echo "Max Images: $MAX_IMAGES"
else
    echo "Max Images: All"
fi
echo "Batch Size: $BATCH_SIZE"
echo "=================================================="

if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory does not exist: $IMAGE_DIR"
    echo "Usage: bash $0 [image_dir] [lpips_net] [max_images] [batch_size]"
    echo "Example: bash $0 generate_data/hazelnut/full/image alex 100 32"
    echo "LPIPS networks: alex, vgg, squeeze"
    exit 1
fi

MAX_IMAGES_ARG=""
if [ -n "$MAX_IMAGES" ]; then
    MAX_IMAGES_ARG="--max_images $MAX_IMAGES"
fi

.env/bin/python eval/compute_ic_lpips.py \
    --image_dir "$IMAGE_DIR" \
    --lpips_net "$LPIPS_NET" \
    --batch_size "$BATCH_SIZE" \
    $MAX_IMAGES_ARG

echo ""
echo "Evaluation completed!"