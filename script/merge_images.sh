#!/bin/bash

# 檢查參數
if [ "$#" -ne 1 ]; then
    echo "使用方式: bash script/merge_images.sh <圖片目錄路徑>"
    echo "範例: bash script/merge_images.sh generate_data/hazelnut/hole/image"
    exit 1
fi

IMAGE_DIR="$1"

# 檢查目錄是否存在
if [ ! -d "$IMAGE_DIR" ]; then
    echo "錯誤: 目錄 '$IMAGE_DIR' 不存在"
    exit 1
fi

echo "處理目錄: $IMAGE_DIR"

# 調用 merge_images.py
.env/bin/python src/merge_images.py "$IMAGE_DIR"

echo "完成！"