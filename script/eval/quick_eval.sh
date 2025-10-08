#!/bin/bash

# 快速評估單一方法的指標

if [ -z "$1" ]; then
    echo "用法: $0 <方法名稱>"
    echo ""
    echo "範例:"
    echo "  $0 stage1-hole-dual"
    echo "  $0 stage2-hole"
    echo "  $0 stage2-hole-attend_excite"
    exit 1
fi

METHOD=$1
MVTEC_NAME="hazelnut"
IMAGE_DIR="generate_data/$MVTEC_NAME/$METHOD/image"

echo "========================================="
echo "快速評估: $METHOD"
echo "========================================="
echo "圖片目錄: $IMAGE_DIR"
echo ""

# 檢查目錄是否存在
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ 錯誤: 目錄不存在"
    echo "   $IMAGE_DIR"
    exit 1
fi

# 計算圖片數量
NUM_IMAGES=$(ls -1 "$IMAGE_DIR"/*.png 2>/dev/null | wc -l)
echo "📊 圖片數量: $NUM_IMAGES"
echo ""

if [ $NUM_IMAGES -eq 0 ]; then
    echo "❌ 錯誤: 沒有找到圖片"
    exit 1
fi

# 顯示前幾張圖片的路徑
echo "📁 範例圖片:"
ls -1 "$IMAGE_DIR"/*.png 2>/dev/null | head -3
echo ""

# 計算 IC-LPIPS
echo "-----------------------------------"
echo "計算 IC-LPIPS (越低越好)..."
echo "-----------------------------------"
.env/bin/python eval/compute_ic_lpips.py \
    --image_dir "$IMAGE_DIR" \
    --batch_size 32

echo ""
echo ""

# 計算 Inception Score
echo "-----------------------------------"
echo "計算 Inception Score (越高越好)..."
echo "-----------------------------------"
.env/bin/python eval/compute_is.py \
    --image_dir "$IMAGE_DIR" \
    --batch_size 32 \
    --splits 10

echo ""
echo "========================================="
echo "評估完成！"
echo "========================================="
