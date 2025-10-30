#!/bin/bash

# IC-LPIPS (Intra-cluster pairwise LPIPS) Evaluation Script
# 使用 DualAnoDiff 的評估方法 - 更準確的 IC-LPIPS 計算
# bash script/eval/eval_ic_lpips.sh hazelnut

SAMPLE_NAME="${1:-hazelnut}"
GENERATE_DATA_PATH="${2:-generate_data}"
MVTEC_PATH="${3:-datasets/mvtec_ad}"
OUTPUT="${4:-ic_lpips_results.csv}"

echo "=================================================="
echo "IC-LPIPS Evaluation (DualAnoDiff Method)"
echo "=================================================="
echo "Sample Name: $SAMPLE_NAME"
echo "Generate Data Path: $GENERATE_DATA_PATH"
echo "MVTec Path: $MVTEC_PATH"
echo "Output File: $OUTPUT"
echo "=================================================="

if [ ! -d "$GENERATE_DATA_PATH/$SAMPLE_NAME" ]; then
    echo "Error: Sample directory does not exist: $GENERATE_DATA_PATH/$SAMPLE_NAME"
    echo "Usage: bash $0 [sample_name] [generate_data_path] [mvtec_path] [output]"
    echo "Example: bash $0 hazelnut generate_data datasets/mvtec_ad ic_lpips_results.csv"
    echo "Sample names: hazelnut, capsule, bottle, etc. (or 'all' for all samples)"
    exit 1
fi

.env/bin/python eval/compute-ic-lpips.py \
    --sample_name "$SAMPLE_NAME" \
    --generate_data_path "$GENERATE_DATA_PATH" \
    --mvtec_path "$MVTEC_PATH" \
    --output "$OUTPUT"

echo ""
echo "Evaluation completed! Results saved to $OUTPUT"
