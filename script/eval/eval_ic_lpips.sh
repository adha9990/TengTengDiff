#!/bin/bash

# IC-LPIPS 評估腳本
# 評估生成圖像的多樣性

echo "========================================"
echo "執行 IC-LPIPS 評估 (多樣性評估)"
echo "========================================"

# 設定參數
SCRIPT_DIR="$(dirname "$0")"
BASE_DIR="$SCRIPT_DIR/../.."
EVAL_DIR="$BASE_DIR/eval"
PYTHON_BIN="$BASE_DIR/.env/bin/python"

# 處理命令行參數
SAMPLE_NAME=${1:-all}
GENERATE_DATA_PATH=${2:-$BASE_DIR/generate_data}
MVTEC_PATH=${3:-$BASE_DIR/datasets/mvtec_ad}
OUTPUT_FILE=${4:-$BASE_DIR/eval_result/ic_lpips_results.csv}

echo "參數設定:"
echo "  樣本名稱: $SAMPLE_NAME"
echo "  生成資料路徑: $GENERATE_DATA_PATH"
echo "  MVTec 路徑: $MVTEC_PATH"
echo "  輸出檔案: $OUTPUT_FILE"

# 檢查必要套件
echo ""
echo "檢查必要套件..."
$PYTHON_BIN -c "import lpips" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安裝 lpips 套件..."
    $PYTHON_BIN -m pip install lpips
fi

# 執行評估
echo ""
echo "開始 IC-LPIPS 評估..."
cd $BASE_DIR && $PYTHON_BIN eval/compute-ic-lpipis.py \
    --sample_name $SAMPLE_NAME \
    --generate_data_path $GENERATE_DATA_PATH \
    --mvtec_path $MVTEC_PATH \
    --output_file $OUTPUT_FILE

if [ $? -eq 0 ]; then
    echo ""
    echo "IC-LPIPS 評估完成！"
    echo "結果檔案: $OUTPUT_FILE"
    
    # 顯示結果摘要
    if [ -f "$OUTPUT_FILE" ]; then
        echo ""
        echo "結果摘要:"
        grep "average" $OUTPUT_FILE | tail -5
    fi
else
    echo ""
    echo "IC-LPIPS 評估失敗"
fi