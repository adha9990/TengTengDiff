#!/bin/bash

# Inception Score Evaluation Script
# 使用 DualAnoDiff 的評估方法 - 使用 torch-fidelity 庫計算 IS
# bash script/eval/eval_inception_score.sh hazelnut

SAMPLE_NAME="${1:-hazelnut}"
GENERATE_DATA_PATH="${2:-generate_data}"
OUTPUT="${3:-IS_results.csv}"
GPU="${4:-0}"

echo "=================================================="
echo "Inception Score Evaluation (DualAnoDiff Method)"
echo "=================================================="
echo "Sample Name: $SAMPLE_NAME"
echo "Generate Data Path: $GENERATE_DATA_PATH"
echo "Output File: $OUTPUT"
echo "GPU: $GPU"
echo "=================================================="

if [ ! -d "$GENERATE_DATA_PATH/$SAMPLE_NAME" ]; then
    echo "Error: Sample directory does not exist: $GENERATE_DATA_PATH/$SAMPLE_NAME"
    echo "Usage: bash $0 [sample_name] [generate_data_path] [output] [gpu]"
    echo "Example: bash $0 hazelnut generate_data IS_results.csv 0"
    echo "Sample names: hazelnut, capsule, bottle, etc. (or 'all' for all samples)"
    exit 1
fi

.env/bin/python eval/compute-is.py \
    --sample_name "$SAMPLE_NAME" \
    --generate_data_path "$GENERATE_DATA_PATH" \
    --output "$OUTPUT" \
    --gpu "$GPU"

echo ""
echo "Evaluation completed! Results saved to $OUTPUT"
