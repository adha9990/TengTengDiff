#!/bin/bash

# 批量執行評估腳本
# 評估 generate_data_dino.2k_sme-5kdino/hazelnut 下所有異常類型的所有 checkpoint

BASE_DIR="/home/bluestar/research/TengTengDiff"
GENERATE_DIR="${BASE_DIR}/generate_data.2.5k_dino-5k_sme/hazelnut"
MVTEC_PATH="${BASE_DIR}/datasets/mvtec_ad"
GPU="${1:-0}"

# 異常類別列表
ANOMALIES=("crack" "print" "hole" "cut")

# Checkpoint 列表
CHECKPOINTS=(1000 2000 3000 4000 5000)

# 計數器
total_count=0
success_count=0
fail_count=0

echo "========================================="
echo "批量執行評估 - generate_data_dino.1"
echo "========================================="
echo "生成數據目錄: ${GENERATE_DIR}"
echo "MVTec 路徑: ${MVTEC_PATH}"
echo "GPU: ${GPU}"
echo ""

# 遍歷所有異常類別
for anomaly in "${ANOMALIES[@]}"; do
    echo "========================================="
    echo ">>> 處理異常類別: ${anomaly}"
    echo "========================================="
    echo ""

    # 遍歷所有 checkpoint
    for checkpoint in "${CHECKPOINTS[@]}"; do
        IMAGE_DIR="${GENERATE_DIR}/stage1-${anomaly}-dual/checkpoint-${checkpoint}/image"

        # 檢查目錄是否存在
        if [ ! -d "$IMAGE_DIR" ]; then
            echo "  [跳過] 目錄不存在: $IMAGE_DIR"
            continue
        fi

        echo "  [${total_count}] 評估: stage1-${anomaly}-dual/checkpoint-${checkpoint}"
        echo "  圖片路徑: $IMAGE_DIR"

        # 執行評估腳本
        if bash "${BASE_DIR}/script/eval/eval_all.sh" "$IMAGE_DIR" "$MVTEC_PATH" "$GPU"; then
            echo "  ✓ 成功"
            ((success_count++))
        else
            echo "  ✗ 失敗"
            ((fail_count++))
        fi

        ((total_count++))
        echo ""
    done

    echo ""
done

echo "========================================="
echo "批量評估完成"
echo "========================================="
echo "總計: ${total_count}"
echo "成功: ${success_count}"
echo "失敗: ${fail_count}"
echo "========================================="
echo ""
echo "評估結果已保存到各個 checkpoint 目錄下:"
echo "  - checkpoint-xxx/ic_lpips_results.csv"
echo "  - checkpoint-xxx/IS_results.csv"
echo "========================================="
echo ""
echo "========================================="
echo "開始生成彙總報告..."
echo "========================================="

# 執行彙總腳本（使用命令列參數）
if "${BASE_DIR}/.env/bin/python" "${BASE_DIR}/summarize_results.py" \
    --generate-dir "${GENERATE_DIR}" \
    --anomalies "${ANOMALIES[@]}" \
    --checkpoints "${CHECKPOINTS[@]}"; then
    echo "✓ 彙總報告生成成功"
    echo "報告位置: ${GENERATE_DIR}/hazelnut_evaluation_summary.csv"
else
    echo "✗ 彙總報告生成失敗"
fi

echo "========================================="
