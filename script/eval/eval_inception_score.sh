#!/bin/bash

# Inception Score 評估腳本
# 評估生成圖像的品質

echo "========================================"
echo "執行 Inception Score 評估 (圖像品質評估)"
echo "========================================"

# 設定參數
SCRIPT_DIR="$(dirname "$0")"
BASE_DIR="$SCRIPT_DIR/../.."
EVAL_DIR="$BASE_DIR/eval"
PYTHON_BIN="$BASE_DIR/.env/bin/python"

# 處理命令行參數
SAMPLE_NAME=${1:-all}
GENERATE_DATA_PATH=${2:-$BASE_DIR/generate_data}
OUTPUT_FILE=${3:-$BASE_DIR/eval_result/inception_score_results.csv}
GPU_ID=${4:-0}

echo "參數設定:"
echo "  樣本名稱: $SAMPLE_NAME"
echo "  生成資料路徑: $GENERATE_DATA_PATH"
echo "  輸出檔案: $OUTPUT_FILE"
echo "  GPU ID: $GPU_ID"

# 檢查必要套件
echo ""
echo "檢查必要套件..."
$PYTHON_BIN -c "import torch_fidelity" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安裝 torch-fidelity 套件..."
    $PYTHON_BIN -m pip install torch-fidelity
fi

# 執行評估
echo ""
echo "開始 Inception Score 評估..."
cd $BASE_DIR && $PYTHON_BIN eval/compute-is.py \
    --sample_name $SAMPLE_NAME \
    --generate_data_path $GENERATE_DATA_PATH \
    --output_file $OUTPUT_FILE \
    --gpu $GPU_ID

if [ $? -eq 0 ]; then
    echo ""
    echo "Inception Score 評估完成！"
    echo "結果檔案: $OUTPUT_FILE"
    
    # 顯示結果摘要
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "結果摘要:"
        grep "average" $OUTPUT_FILE | tail -5
    fi
else
    echo ""
    echo "Inception Score 評估失敗"
fi