#!/bin/bash

# 定位評估完整流程
# 包含訓練和測試兩個步驟

echo "========================================"
echo "定位評估完整流程"
echo "========================================"

# 設定參數
SCRIPT_DIR="$(dirname "$0")"

# 處理命令行參數
SAMPLE_NAME=${1:-all}
MVTEC_PATH=${2:-}
GPU_ID=${3:-0}
SKIP_TRAIN=${4:-false}

echo "使用方式: $0 [sample_name] [mvtec_path] [gpu_id] [skip_train]"
echo ""
echo "參數:"
echo "  樣本名稱: $SAMPLE_NAME"
echo "  MVTec 路徑: $MVTEC_PATH"
echo "  GPU ID: $GPU_ID"
echo "  跳過訓練: $SKIP_TRAIN"

# 執行訓練（如果需要）
if [ "$SKIP_TRAIN" != "true" ]; then
    echo ""
    echo "步驟 1/2: 訓練定位模型"
    echo "----------------------------------------"
    if [ -z "$MVTEC_PATH" ]; then
        bash $SCRIPT_DIR/train_localization.sh $SAMPLE_NAME
    else
        bash $SCRIPT_DIR/train_localization.sh $SAMPLE_NAME $MVTEC_PATH
    fi
    
    if [ $? -ne 0 ]; then
        echo "定位模型訓練失敗，終止流程"
        exit 1
    fi
else
    echo ""
    echo "跳過訓練步驟"
fi

# 執行測試
echo ""
echo "步驟 2/2: 測試定位模型"
echo "----------------------------------------"
if [ -z "$MVTEC_PATH" ]; then
    bash $SCRIPT_DIR/test_localization.sh $SAMPLE_NAME
else
    bash $SCRIPT_DIR/test_localization.sh $SAMPLE_NAME $MVTEC_PATH $GPU_ID
fi

echo ""
echo "========================================"
echo "定位評估流程完成！"
echo "========================================"