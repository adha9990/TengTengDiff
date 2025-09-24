#!/bin/bash

# 執行所有生成評估指標的整合腳本
# Usage: bash script/eval/eval_all.sh [image_dir] [lpips_net] [max_images]

IMAGE_DIR="${1:-generate_data/hazelnut/full/image}"
LPIPS_NET="${2:-alex}"
MAX_IMAGES="${3:-}"
BATCH_SIZE="${4:-32}"
SPLITS="${5:-10}"

echo "=========================================="
echo "    生成圖片評估指標 - 整合腳本"
echo "=========================================="
echo "圖片目錄: $IMAGE_DIR"
echo "LPIPS 網路: $LPIPS_NET"
if [ -n "$MAX_IMAGES" ]; then
    echo "最大圖片數: $MAX_IMAGES"
else
    echo "最大圖片數: 全部"
fi
echo "批次大小: $BATCH_SIZE"
echo "IS Splits: $SPLITS"
echo "=========================================="

# 檢查目錄是否存在
if [ ! -d "$IMAGE_DIR" ]; then
    echo "錯誤: 圖片目錄不存在: $IMAGE_DIR"
    echo ""
    echo "使用方法: bash $0 [image_dir] [lpips_net] [max_images] [batch_size] [splits]"
    echo "範例: bash $0 generate_data/hazelnut/full/image alex 100 32 10"
    echo ""
    echo "參數說明:"
    echo "  image_dir  : 圖片目錄路徑 (預設: generate_data/hazelnut/full/image)"
    echo "  lpips_net  : LPIPS 網路 [alex/vgg/squeeze] (預設: alex)"
    echo "  max_images : 最大圖片數量 (預設: 全部)"
    echo "  batch_size : 批次大小 (預設: 32)"
    echo "  splits     : IS 計算的分割數 (預設: 10)"
    exit 1
fi

# 取得輸出目錄（圖片目錄的上一層，處理結尾斜線）
IMAGE_DIR_CLEAN=${IMAGE_DIR%/}
OUTPUT_DIR=$(dirname "$IMAGE_DIR_CLEAN")
echo ""
echo "評估結果將保存到: $OUTPUT_DIR"
echo ""

# 1. 執行 IC-LPIPS 評估
echo "=========================================="
echo "步驟 1/2: 執行 IC-LPIPS 評估"
echo "=========================================="

MAX_IMAGES_ARG=""
if [ -n "$MAX_IMAGES" ]; then
    MAX_IMAGES_ARG="--max_images $MAX_IMAGES"
fi

.env/bin/python eval/compute_ic_lpips.py \
    --image_dir "$IMAGE_DIR" \
    --lpips_net "$LPIPS_NET" \
    --batch_size "$BATCH_SIZE" \
    $MAX_IMAGES_ARG

if [ $? -ne 0 ]; then
    echo "錯誤: IC-LPIPS 評估失敗"
    exit 1
fi

echo ""
echo "IC-LPIPS 評估完成！結果已保存到: $OUTPUT_DIR/ic_lpips.txt"
echo ""

# 2. 執行 Inception Score 評估
echo "=========================================="
echo "步驟 2/2: 執行 Inception Score 評估"
echo "=========================================="

.env/bin/python eval/compute_is.py \
    --image_dir "$IMAGE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --splits "$SPLITS"

if [ $? -ne 0 ]; then
    echo "錯誤: Inception Score 評估失敗"
    exit 1
fi

echo ""
echo "Inception Score 評估完成！結果已保存到: $OUTPUT_DIR/inception_score.txt"
echo ""

# 顯示所有結果摘要
echo "=========================================="
echo "          評估結果摘要"
echo "=========================================="
echo ""
echo "IC-LPIPS 結果:"
if [ -f "$OUTPUT_DIR/ic_lpips.txt" ]; then
    cat "$OUTPUT_DIR/ic_lpips.txt" | grep -E "Mean|Median"
else
    echo "  檔案未找到: $OUTPUT_DIR/ic_lpips.txt"
fi

echo ""
echo "Inception Score 結果:"
if [ -f "$OUTPUT_DIR/inception_score.txt" ]; then
    cat "$OUTPUT_DIR/inception_score.txt" | grep "Inception Score"
else
    echo "  檔案未找到: $OUTPUT_DIR/inception_score.txt"
fi

echo ""
echo "=========================================="
echo "         所有評估完成！"
echo "=========================================="
echo ""
echo "詳細結果已保存到:"
echo "  - $OUTPUT_DIR/ic_lpips.txt"
echo "  - $OUTPUT_DIR/inception_score.txt"
echo ""