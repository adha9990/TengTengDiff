#!/bin/bash

# 定位模型訓練腳本
# 訓練用於評估瑕疵定位準確性的 U-Net 模型

echo "========================================"
echo "訓練定位評估模型"
echo "========================================"

# 設定參數
SCRIPT_DIR="$(dirname "$0")"
BASE_DIR="$SCRIPT_DIR/../.."
EVAL_DIR="$BASE_DIR/eval"
PYTHON_BIN="$BASE_DIR/.env/bin/python"

# 處理命令行參數
SAMPLE_NAME=${1:-all}
MVTEC_PATH=${2:-$BASE_DIR/datasets/mvtec_ad}
GENERATED_DATA_PATH=${3:-$BASE_DIR/generate_data}
GPU_ID=${4:-0}
CHECKPOINT_PATH=${5:-$BASE_DIR/eval_result/localization_checkpoints}

echo "參數設定:"
echo "  樣本名稱: $SAMPLE_NAME"
echo "  MVTec 路徑: $MVTEC_PATH"
echo "  生成資料路徑: $GENERATED_DATA_PATH"
echo "  GPU ID: $GPU_ID"
echo "  檢查點路徑: $CHECKPOINT_PATH"

# 創建檢查點目錄
mkdir -p $CHECKPOINT_PATH

# 執行訓練
echo ""
echo "開始訓練定位模型..."

cd $BASE_DIR && $PYTHON_BIN eval/train-localization.py \
    --gpu_id $GPU_ID \
    --sample_name $SAMPLE_NAME \
    --data_path $MVTEC_PATH \
    --generated_data_path $GENERATED_DATA_PATH \
    --checkpoint_path $CHECKPOINT_PATH

if [ $? -eq 0 ]; then
    echo ""
    echo "定位模型訓練完成！"
    echo "檢查點保存於: $CHECKPOINT_PATH"
else
    echo ""
    echo "定位模型訓練失敗"
fi