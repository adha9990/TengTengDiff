#!/bin/bash

# 定位評估測試腳本
# 評估瑕疵定位的準確性

echo "========================================"
echo "執行定位評估 (瑕疵定位準確性)"
echo "========================================"

# 設定參數
SCRIPT_DIR="$(dirname "$0")"
BASE_DIR="$SCRIPT_DIR/../.."
EVAL_DIR="$BASE_DIR/eval"
PYTHON_BIN="$BASE_DIR/.env/bin/python"

# 處理命令行參數
SAMPLE_NAME=${1:-all}
MVTEC_PATH=${2:-$BASE_DIR/datasets/mvtec_ad}
GPU_ID=${3:-0}
CHECKPOINT_PATH=${4:-$BASE_DIR/eval_result/localization_checkpoints}
OUTPUT_DIR=${5:-$BASE_DIR/eval_result/localization_results}

echo "參數設定:"
echo "  樣本名稱: $SAMPLE_NAME"
echo "  MVTec 路徑: $MVTEC_PATH"
echo "  GPU ID: $GPU_ID"
echo "  檢查點路徑: $CHECKPOINT_PATH"
echo "  輸出目錄: $OUTPUT_DIR"

# 創建輸出目錄
mkdir -p $OUTPUT_DIR

# 執行測試
echo ""
echo "開始定位評估..."

cd $BASE_DIR && $PYTHON_BIN eval/test-localization.py \
    --gpu_id $GPU_ID \
    --sample_name $SAMPLE_NAME \
    --mvtec_path $MVTEC_PATH \
    --checkpoint_path $CHECKPOINT_PATH

if [ $? -eq 0 ]; then
    echo ""
    echo "定位評估完成！"
    
    # 移動結果檔案
    if [ -f "eval/result.csv" ]; then
        mv eval/result.csv $OUTPUT_DIR/localization_metrics.csv
        echo "評估結果: $OUTPUT_DIR/localization_metrics.csv"
    fi
    
    # 移動視覺化結果
    if [ -d "eval/result" ]; then
        mv eval/result $OUTPUT_DIR/visualizations
        echo "視覺化結果: $OUTPUT_DIR/visualizations/"
    fi
else
    echo ""
    echo "定位評估失敗"
fi