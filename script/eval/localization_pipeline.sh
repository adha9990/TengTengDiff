#!/bin/bash

# 異常定位完整流程 - 訓練 + 測試
# Usage: bash script/eval/localization_pipeline.sh [sample_name] [generated_data_path] [mvtec_path]

SAMPLE_NAME="${1:-hazelnut}"
GENERATED_DATA_PATH="${2:-generate_data}"
MVTEC_PATH="${3:-datasets/mvtec_ad}"
CHECKPOINT_PATH="${4:-checkpoints/localization}"
BATCH_SIZE="${5:-16}"
LEARNING_RATE="${6:-0.0001}"
EPOCHS="${7:-200}"
GPU_ID="${8:-0}"

echo "=========================================="
echo "   異常定位完整流程 (Train + Test)"
echo "=========================================="
echo "樣本名稱: $SAMPLE_NAME"
echo "生成數據路徑: $GENERATED_DATA_PATH"
echo "MVTec 路徑: $MVTEC_PATH"
echo "Checkpoint 路徑: $CHECKPOINT_PATH"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Epochs: $EPOCHS"
echo "GPU ID: $GPU_ID"
echo "=========================================="

# 檢查目錄
if [ ! -d "$GENERATED_DATA_PATH/$SAMPLE_NAME" ]; then
    echo "錯誤: 生成數據目錄不存在: $GENERATED_DATA_PATH/$SAMPLE_NAME"
    echo ""
    echo "使用方法: bash $0 [sample_name] [generated_data_path] [mvtec_path] [checkpoint_path] [batch_size] [lr] [epochs] [gpu]"
    echo "範例: bash $0 hazelnut generate_data datasets/mvtec_ad checkpoints/localization 16 0.0001 200 0"
    echo ""
    echo "參數說明:"
    echo "  sample_name          : 樣本名稱 (預設: hazelnut, 可用 'all' 處理所有樣本)"
    echo "  generated_data_path  : 生成數據路徑 (預設: generate_data)"
    echo "  mvtec_path           : MVTec 數據集路徑 (預設: datasets/mvtec_ad)"
    echo "  checkpoint_path      : 模型保存路徑 (預設: checkpoints/localization)"
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

# 步驟 1: 訓練定位模型
echo ""
echo "=========================================="
echo "步驟 1/2: 訓練異常定位模型"
echo "=========================================="
echo ""

bash script/eval/train_localization.sh \
    "$SAMPLE_NAME" \
    "$GENERATED_DATA_PATH" \
    "$MVTEC_PATH" \
    "$CHECKPOINT_PATH" \
    "$BATCH_SIZE" \
    "$LEARNING_RATE" \
    "$EPOCHS" \
    "$GPU_ID"

if [ $? -ne 0 ]; then
    echo ""
    echo "錯誤: 訓練失敗"
    exit 1
fi

echo ""
echo "訓練完成！模型已保存到: $CHECKPOINT_PATH"
echo ""

# 步驟 2: 測試定位模型
echo ""
echo "=========================================="
echo "步驟 2/2: 測試異常定位模型"
echo "=========================================="
echo ""

bash script/eval/test_localization.sh \
    "$SAMPLE_NAME" \
    "$MVTEC_PATH" \
    "$CHECKPOINT_PATH" \
    "$GPU_ID"

if [ $? -ne 0 ]; then
    echo ""
    echo "錯誤: 測試失敗"
    exit 1
fi

echo ""
echo "=========================================="
echo "         所有步驟完成！"
echo "=========================================="
echo ""
echo "結果文件:"
echo "  - $CHECKPOINT_PATH/         (訓練的模型)"
echo "  - result.csv                (測試結果指標)"
echo "  - result/                   (視覺化結果)"
echo ""

# 顯示最終結果
if [ -f "result.csv" ]; then
    echo "最終評估結果:"
    echo ""
    cat result.csv
    echo ""
fi

echo "=========================================="
echo ""
