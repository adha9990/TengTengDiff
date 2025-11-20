#!/bin/bash

# 批量執行 merge_images.sh 腳本
# 處理 generate_data/hazelnut 下所有類別的所有 checkpoint

BASE_DIR="/home/bluestar/research/TengTengDiff"
GENERATE_DIR="${BASE_DIR}/generate_data.2.5k_dino-5k_sme/hazelnut"

# 異常類別列表
ANOMALIES=("crack" "print" "hole" "cut")

# Checkpoint 列表
CHECKPOINTS=(1000 2000 3000 4000 5000)

# 計數器
total_count=0
success_count=0
fail_count=0

echo "========================================="
echo "批量執行 merge_images.sh"
echo "========================================="
echo ""

# 遍歷所有異常類別
for anomaly in "${ANOMALIES[@]}"; do
    echo ">>> 處理異常類別: ${anomaly}"
    echo ""

    # 遍歷所有 checkpoint
    for checkpoint in "${CHECKPOINTS[@]}"; do
        IMAGE_DIR="${GENERATE_DIR}/stage1-${anomaly}-dual/checkpoint-${checkpoint}/image"

        # 檢查目錄是否存在
        if [ ! -d "$IMAGE_DIR" ]; then
            echo "  [跳過] 目錄不存在: $IMAGE_DIR"
            continue
        fi

        echo "  [${total_count}] 處理: stage1-${anomaly}-dual/checkpoint-${checkpoint}"

        # 執行 merge_images.sh
        if bash "${BASE_DIR}/script/merge_images.sh" "$IMAGE_DIR"; then
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
echo "批量處理完成"
echo "========================================="
echo "總計: ${total_count}"
echo "成功: ${success_count}"
echo "失敗: ${fail_count}"
echo "========================================="
