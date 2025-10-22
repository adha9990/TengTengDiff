#!/bin/bash

# 完整分割管線腳本
# 從原始圖片生成遮罩，然後使用遮罩處理圖片保留中間物體
#
# 使用方式: ./script/segment_pipeline.sh <輸入目錄> [背景類型]
#
# 輸出:
#   - {輸入目錄}_mask: U-2-Net 生成的遮罩
#   - {輸入目錄}_segmented: 使用遮罩處理後的圖片

export MODEL_NAME="models/u2net.pth"

# 從命令列參數讀取
IMAGE_DIR="$1"
BACKGROUND="${2:-transparent}"

# 顯示使用說明
if [ $# -eq 0 ]; then
    echo "=========================================="
    echo "  完整分割管線腳本"
    echo "=========================================="
    echo ""
    echo "使用方式: $0 <輸入目錄> [背景類型]"
    echo ""
    echo "參數說明:"
    echo "  輸入目錄: 原始圖片所在目錄"
    echo "  背景類型: (可選) transparent/white/black，預設為 transparent"
    echo ""
    echo "輸出:"
    echo "  - {輸入目錄}_mask: U-2-Net 生成的遮罩"
    echo "  - {輸入目錄}_segmented: 使用遮罩處理後的圖片"
    echo ""
    echo "範例:"
    echo "  $0 datasets/mvtec_ad/hazelnut/train/good"
    echo "  $0 datasets/mvtec_ad/hazelnut/train/good white"
    echo ""
    exit 1
fi

# 檢查輸入目錄是否存在
if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ 錯誤: 目錄 '$IMAGE_DIR' 不存在"
    exit 1
fi

# 設定輸出目錄
FOLDER_NAME=$(basename "$IMAGE_DIR")
PARENT_DIR=$(dirname "$IMAGE_DIR")
MASK_DIR="$PARENT_DIR/${FOLDER_NAME}_mask"
SEGMENTED_DIR="$PARENT_DIR/${FOLDER_NAME}_segmented"

echo "=========================================="
echo "  完整分割管線"
echo "=========================================="
echo "📁 輸入目錄: $IMAGE_DIR"
echo "🎭 遮罩輸出: $MASK_DIR"
echo "🖼️  最終輸出: $SEGMENTED_DIR"
echo "🎨 背景類型: $BACKGROUND"
echo ""

# 步驟 1: 使用 U-2-Net 生成遮罩
echo "=========================================="
echo "  步驟 1/2: 生成遮罩 (U-2-Net)"
echo "=========================================="

# 檢查並修改目錄權限（如果需要）
if [ ! -w "$PARENT_DIR" ]; then
    echo "⚠️  修改父目錄權限以允許寫入..."
    chmod u+w "$PARENT_DIR"
fi

# 建立遮罩輸出目錄
mkdir -p "$MASK_DIR"

.env/bin/python extra_repository/U-2-Net/u2net_test.py \
    --model_dir=$MODEL_NAME \
    --input_dir="$IMAGE_DIR" \
    --output_dir="$MASK_DIR"

# 檢查 U-2-Net 是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ U-2-Net 遮罩生成失敗"
    exit 1
fi

echo ""
echo "✅ 遮罩生成完成！"
echo ""

# 步驟 2: 使用遮罩處理原始圖片
echo "=========================================="
echo "  步驟 2/2: 應用遮罩到原始圖片"
echo "=========================================="

.env/bin/python apply_mask.py "$IMAGE_DIR" "$MASK_DIR" "$SEGMENTED_DIR" "$BACKGROUND"

# 檢查是否成功
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 遮罩應用失敗"
    exit 1
fi

echo ""
echo "=========================================="
echo "  ✅ 完整管線執行完成！"
echo "=========================================="
echo "🎭 遮罩位置: $MASK_DIR"
echo "🖼️  分割圖片: $SEGMENTED_DIR"
echo "=========================================="
