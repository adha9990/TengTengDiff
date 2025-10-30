#!/bin/bash

# 測試異常定位模型 - DualAnoDiff 方法
# 評估指標: AUROC, AP, F1-max (Image & Pixel level), PRO (Pixel level)
# Usage: bash script/eval/test_localization.sh [sample_name] [mvtec_path] [checkpoint_path]

SAMPLE_NAME="${1:-hazelnut}"
MVTEC_PATH="${2:-datasets/mvtec_ad}"
CHECKPOINT_PATH="${3:-checkpoints/localization}"
GPU_ID="${4:-0}"

echo "=========================================="
echo "   測試異常定位模型 (Localization)"
echo "=========================================="
echo "樣本名稱: $SAMPLE_NAME"
echo "MVTec 路徑: $MVTEC_PATH"
echo "Checkpoint 路徑: $CHECKPOINT_PATH"
echo "GPU ID: $GPU_ID"
echo "=========================================="

# 檢查 MVTec 目錄是否存在
if [ ! -d "$MVTEC_PATH" ]; then
    echo "錯誤: MVTec 數據集目錄不存在: $MVTEC_PATH"
    echo ""
    echo "使用方法: bash $0 [sample_name] [mvtec_path] [checkpoint_path] [gpu]"
    echo "範例: bash $0 hazelnut datasets/mvtec_ad checkpoints/localization 0"
    echo ""
    echo "參數說明:"
    echo "  sample_name      : 樣本名稱 (預設: hazelnut, 可用 'all' 測試所有樣本)"
    echo "  mvtec_path       : MVTec 數據集路徑 (預設: datasets/mvtec_ad)"
    echo "  checkpoint_path  : 模型 checkpoint 路徑 (預設: checkpoints/localization)"
    echo "  gpu              : GPU ID (預設: 0)"
    exit 1
fi

# 檢查 checkpoint 目錄是否存在
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "警告: Checkpoint 目錄不存在: $CHECKPOINT_PATH"
    echo "請先運行 train_localization.sh 訓練模型"
    exit 1
fi

echo ""
echo "開始測試..."
echo ""

.env/bin/python eval/test-localization.py \
    --sample_name "$SAMPLE_NAME" \
    --mvtec_path "$MVTEC_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --gpu_id "$GPU_ID"

if [ $? -ne 0 ]; then
    echo ""
    echo "錯誤: 測試失敗"
    exit 1
fi

echo ""
echo "=========================================="
echo "         測試完成！"
echo "=========================================="
echo ""
echo "結果已保存到:"
echo "  - result.csv         (評估指標)"
echo "  - result/            (視覺化結果)"
echo ""

# 顯示結果摘要
if [ -f "result.csv" ]; then
    echo "評估結果摘要:"
    cat result.csv
    echo ""
fi
