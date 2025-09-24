#!/usr/bin/env python3

import os
import numpy as np
from PIL import Image
import math
from pathlib import Path

def merge_images_to_grid(image_dir, output_name="merged_preview.png", max_size_per_image=(256, 256)):
    image_dir = Path(image_dir)

    image_files = sorted([f for f in image_dir.glob("*.png") if f.name != output_name])

    if not image_files:
        print("沒有找到圖片文件")
        return

    print(f"找到 {len(image_files)} 張圖片")

    images = []
    for img_path in image_files:
        img = Image.open(img_path)
        img.thumbnail(max_size_per_image, Image.Resampling.LANCZOS)
        images.append(img)

    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)

    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    grid_width = cols * max_width
    grid_height = rows * max_height

    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * max_width + (max_width - img.width) // 2
        y = row * max_height + (max_height - img.height) // 2
        grid_image.paste(img, (x, y))

    output_path = image_dir / output_name
    grid_image.save(output_path)
    print(f"合併圖片已儲存至: {output_path}")
    print(f"網格大小: {cols} x {rows}")
    print(f"總圖片尺寸: {grid_width} x {grid_height} pixels")

    return output_path

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方式: python merge_images.py <圖片目錄路徑>")
        print("範例: python merge_images.py ./generate_data/hazelnut/hole/image")
        sys.exit(1)

    image_directory = Path(sys.argv[1])

    # 保存到上層目錄
    parent_dir = image_directory.parent
    dir_name = image_directory.name
    output_name = f"{dir_name}_merged.png"

    # 調用函數，但將輸出保存到上層目錄
    merge_images_to_grid(image_directory, output_name=output_name)

    # 移動檔案到上層目錄
    src_path = image_directory / output_name
    dst_path = parent_dir / output_name
    if src_path.exists():
        src_path.rename(dst_path)
        print(f"檔案已移動至: {dst_path}")