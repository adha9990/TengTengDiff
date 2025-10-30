#!/bin/bash

# 訓練異常定位模型 - DualAnoDiff 方法
# Usage: bash script/eval/train_localization.sh [sample_name] [generated_data_path] [mvtec_path]

SAMPLE_NAME="${1:-hazelnut}"
GENERATED_DATA_PATH="${2:-generate_data}"
MVTEC_PATH="${3:-datasets/mvtec_ad}"
SAVE_PATH="${4:-checkpoints/localization}"
BATCH_SIZE="${5:-16}"
LEARNING_RATE="${6:-0.0001}"
EPOCHS="${7:-200}"
GPU_ID="${8:-0}"

echo "=========================================="
echo "   訓練異常定位模型 (Localization)"
echo "=========================================="
echo "樣本名稱: $SAMPLE_NAME"
echo "生成數據路徑: $GENERATED_DATA_PATH"
echo "MVTec 路徑: $MVTEC_PATH"
echo "保存路徑: $SAVE_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "GPU ID: $GPU_ID"
echo "=========================================="

# 檢查目錄是否存在
if [ ! -d "$GENERATED_DATA_PATH/$SAMPLE_NAME" ]; then
    echo "錯誤: 生成數據目錄不存在: $GENERATED_DATA_PATH/$SAMPLE_NAME"
    echo ""
    echo "使用方法: bash $0 [sample_name] [generated_data_path] [mvtec_path] [save_path] [batch_size] [lr] [epochs] [gpu]"
    echo "範例: bash $0 hazelnut generate_data datasets/mvtec_ad checkpoints/localization 16 0.0001 200 0"
    echo ""
    echo "參數說明:"
    echo "  sample_name          : 樣本名稱 (預設: hazelnut, 可用 'all' 訓練所有樣本)"
    echo "  generated_data_path  : 生成數據路徑 (預設: generate_data)"
    echo "  mvtec_path           : MVTec 數據集路徑 (預設: datasets/mvtec_ad)"
    echo "  save_path            : 模型保存路徑 (預設: checkpoints/localization)"
    echo "  batch_size           : 批次大小 (預設: 16)"
    echo "  lr                   : 學習率 (預設: 0.0001)"
    echo "  epochs               : 訓練輪數 (預設: 200)"
    echo "  gpu                  : GPU ID (預設: 0)"
    exit 1
fi

if [ ! -d "$MVTEC_PATH" ]; then
    echo "錯誤: MVTec 數據集目錄不存在: $MVTEC_PATH"
    exit 1
fi

# 創建保存目錄
mkdir -p "$SAVE_PATH"
mkdir -p logs

echo ""
echo "開始訓練..."
echo ""

.env/bin/python eval/train-localization.py \
    --sample_name "$SAMPLE_NAME" \
    --generated_data_path "$GENERATED_DATA_PATH" \
    --mvtec_path "$MVTEC_PATH" \
    --save_path "$SAVE_PATH" \
    --bs "$BATCH_SIZE" \
    --lr "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    --gpu_id "$GPU_ID"

if [ $? -ne 0 ]; then
    echo ""
    echo "錯誤: 訓練失敗"
    exit 1
fi

echo ""
echo "=========================================="
echo "         訓練完成！"
echo "=========================================="
echo ""
echo "模型已保存到: $SAVE_PATH"
echo ""
