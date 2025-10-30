#!/bin/bash

# 執行所有生成評估指標的整合腳本 - DualAnoDiff 方法
# Usage:
#   方式1（原有方式）: bash script/eval/eval_all.sh [sample_name] [generate_data_path] [mvtec_path] [gpu]
#   方式2（直接指定圖片路徑）: bash script/eval/eval_all.sh [image_path] [mvtec_path] [gpu]

# 檢查第一個參數是否是完整路徑（包含 /image 結尾）
if [[ "$1" == *"/image" ]] || [[ "$1" == *"/image/" ]]; then
    # 方式2：直接指定圖片路徑
    IMAGE_PATH="${1%/}"  # 移除尾部斜線
    MVTEC_PATH="${2:-datasets/mvtec_ad}"
    GPU="${3:-0}"

    # 從路徑中解析 sample_name 和 anomaly_name
    # 支援兩種路徑格式:
    #   格式1: .../generate_data/{sample_name}/{anomaly_name}/checkpoint-xxx/image
    #   格式2: .../generate_data3/{sample_name}/{anomaly_name}/image
    IMAGE_PATH_ABS=$(realpath "$IMAGE_PATH")
    PARENT_DIR=$(dirname "$IMAGE_PATH_ABS")
    PARENT_BASENAME=$(basename "$PARENT_DIR")

    # 檢查父目錄是否為 checkpoint-xxx 格式
    if [[ "$PARENT_BASENAME" == checkpoint-* ]]; then
        # 格式1: 有 checkpoint 層級
        ANOMALY_DIR=$(dirname "$PARENT_DIR")
        SAMPLE_DIR=$(dirname "$ANOMALY_DIR")
        GENERATE_DATA_PATH=$(dirname "$SAMPLE_DIR")

        SAMPLE_NAME=$(basename "$SAMPLE_DIR")
        ANOMALY_NAME=$(basename "$ANOMALY_DIR")/$(basename "$PARENT_DIR")
    else
        # 格式2: 沒有 checkpoint 層級
        ANOMALY_DIR="$PARENT_DIR"
        SAMPLE_DIR=$(dirname "$ANOMALY_DIR")
        GENERATE_DATA_PATH=$(dirname "$SAMPLE_DIR")

        SAMPLE_NAME=$(basename "$SAMPLE_DIR")
        ANOMALY_NAME=$(basename "$ANOMALY_DIR")
    fi

    echo "=========================================="
    echo "    生成圖片評估指標 - 直接路徑模式"
    echo "=========================================="
    echo "圖片路徑: $IMAGE_PATH_ABS"
    echo "樣本名稱: $SAMPLE_NAME"
    echo "異常類型: $ANOMALY_NAME"
    echo "生成數據根路徑: $GENERATE_DATA_PATH"
    echo "MVTec 路徑: $MVTEC_PATH"
    echo "GPU: $GPU"
    echo "=========================================="

    # 檢查圖片目錄是否存在
    if [ ! -d "$IMAGE_PATH_ABS" ]; then
        echo "錯誤: 圖片目錄不存在: $IMAGE_PATH_ABS"
        exit 1
    fi

    # 檢查 MVTec 測試目錄是否存在
    MVTEC_TEST_DIR="$MVTEC_PATH/$SAMPLE_NAME/test"
    if [ ! -d "$MVTEC_TEST_DIR" ]; then
        echo "錯誤: MVTec 測試目錄不存在: $MVTEC_TEST_DIR"
        echo "請確認樣本名稱是否正確"
        exit 1
    fi
else
    # 方式1：原有的目錄結構方式
    SAMPLE_NAME="${1:-hazelnut}"
    GENERATE_DATA_PATH="${2:-generate_data}"
    MVTEC_PATH="${3:-datasets/mvtec_ad}"
    GPU="${4:-0}"

    echo "=========================================="
    echo "    生成圖片評估指標 - DualAnoDiff 方法"
    echo "=========================================="
    echo "樣本名稱: $SAMPLE_NAME"
    echo "生成數據路徑: $GENERATE_DATA_PATH"
    echo "MVTec 路徑: $MVTEC_PATH"
    echo "GPU: $GPU"
    echo "=========================================="

    # 檢查目錄是否存在
    if [ ! -d "$GENERATE_DATA_PATH/$SAMPLE_NAME" ]; then
        echo "錯誤: 樣本目錄不存在: $GENERATE_DATA_PATH/$SAMPLE_NAME"
        echo ""
        echo "使用方法:"
        echo "  方式1: bash $0 [sample_name] [generate_data_path] [mvtec_path] [gpu]"
        echo "  方式2: bash $0 [image_path] [mvtec_path] [gpu]"
        echo ""
        echo "範例:"
        echo "  方式1: bash $0 hazelnut generate_data datasets/mvtec_ad 0"
        echo "  方式2: bash $0 generate_data/hazelnut/stage2-hole-dual/checkpoint-5000/image datasets/mvtec_ad 0"
        echo ""
        echo "參數說明:"
        echo "  方式1 參數:"
        echo "    sample_name         : 樣本名稱 (預設: hazelnut, 可用 'all' 評估所有樣本)"
        echo "    generate_data_path  : 生成數據路徑 (預設: generate_data)"
        echo "    mvtec_path          : MVTec 數據集路徑 (預設: datasets/mvtec_ad)"
        echo "    gpu                 : GPU ID (預設: 0)"
        echo ""
        echo "  方式2 參數:"
        echo "    image_path          : 圖片目錄的完整路徑（必須以 /image 結尾）"
        echo "    mvtec_path          : MVTec 數據集路徑 (預設: datasets/mvtec_ad)"
        echo "    gpu                 : GPU ID (預設: 0)"
        exit 1
    fi

    IMAGE_PATH=""
    ANOMALY_NAME=""
    OUTPUT_DIR="."
fi

# 設定輸出目錄
if [ -n "$IMAGE_PATH" ]; then
    # 直接路徑模式：將結果保存到 image 的父目錄
    # 格式1: checkpoint-xxx 目錄
    # 格式2: stage2-xxx-dual 目錄
    OUTPUT_DIR="$PARENT_DIR"
    echo ""
    echo "評估結果將保存到: $OUTPUT_DIR"
    echo ""
else
    # 原有模式：保存到當前目錄
    OUTPUT_DIR="."
    echo ""
    echo "評估結果將保存到當前目錄"
    echo ""
fi

# 1. 執行 IC-LPIPS 評估
echo "=========================================="
echo "步驟 1/2: 執行 IC-LPIPS 評估"
echo "=========================================="

if [ -n "$IMAGE_PATH" ]; then
    # 直接路徑模式
    .env/bin/python eval/compute-ic-lpips.py \
        --sample_name "$SAMPLE_NAME" \
        --anomaly_name "$ANOMALY_NAME" \
        --image_path "$IMAGE_PATH_ABS" \
        --mvtec_path "$MVTEC_PATH" \
        --output "$OUTPUT_DIR/ic_lpips_results.csv" \
        --direct_path_mode
else
    # 原有目錄結構模式
    .env/bin/python eval/compute-ic-lpips.py \
        --sample_name "$SAMPLE_NAME" \
        --generate_data_path "$GENERATE_DATA_PATH" \
        --mvtec_path "$MVTEC_PATH" \
        --output "$OUTPUT_DIR/ic_lpips_results.csv"
fi

if [ $? -ne 0 ]; then
    echo "錯誤: IC-LPIPS 評估失敗"
    exit 1
fi

echo ""
echo "IC-LPIPS 評估完成！結果已保存到: $OUTPUT_DIR/ic_lpips_results.csv"
echo ""

# 2. 執行 Inception Score 評估
echo "=========================================="
echo "步驟 2/2: 執行 Inception Score 評估"
echo "=========================================="

if [ -n "$IMAGE_PATH" ]; then
    # 直接路徑模式
    .env/bin/python eval/compute-is.py \
        --sample_name "$SAMPLE_NAME" \
        --anomaly_name "$ANOMALY_NAME" \
        --image_path "$IMAGE_PATH_ABS" \
        --output "$OUTPUT_DIR/IS_results.csv" \
        --gpu "$GPU" \
        --direct_path_mode
else
    # 原有目錄結構模式
    .env/bin/python eval/compute-is.py \
        --sample_name "$SAMPLE_NAME" \
        --generate_data_path "$GENERATE_DATA_PATH" \
        --output "$OUTPUT_DIR/IS_results.csv" \
        --gpu "$GPU"
fi

if [ $? -ne 0 ]; then
    echo "錯誤: Inception Score 評估失敗"
    exit 1
fi

echo ""
echo "Inception Score 評估完成！結果已保存到: $OUTPUT_DIR/IS_results.csv"
echo ""

# 顯示所有結果摘要
echo "=========================================="
echo "          評估結果摘要"
echo "=========================================="
echo ""
echo "IC-LPIPS 結果:"
if [ -f "$OUTPUT_DIR/ic_lpips_results.csv" ]; then
    cat "$OUTPUT_DIR/ic_lpips_results.csv"
else
    echo "  檔案未找到: $OUTPUT_DIR/ic_lpips_results.csv"
fi

echo ""
echo "Inception Score 結果:"
if [ -f "$OUTPUT_DIR/IS_results.csv" ]; then
    cat "$OUTPUT_DIR/IS_results.csv"
else
    echo "  檔案未找到: $OUTPUT_DIR/IS_results.csv"
fi

echo ""
echo "=========================================="
echo "         所有評估完成！"
echo "=========================================="
echo ""
echo "詳細結果已保存到:"
echo "  - $OUTPUT_DIR/ic_lpips_results.csv"
echo "  - $OUTPUT_DIR/IS_results.csv"
echo ""
