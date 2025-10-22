#!/usr/bin/env python3
"""
使用遮罩分割圖片，保留物體中間部分
從原始圖片中使用遮罩提取前景物體
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def apply_mask_to_image(image_path, mask_path, output_path, background='transparent'):
    """
    使用遮罩將圖片分割，保留物體部分

    Args:
        image_path: 原始圖片路徑
        mask_path: 遮罩圖片路徑
        output_path: 輸出圖片路徑
        background: 背景類型 ('transparent', 'white', 'black')
    """
    # 讀取原始圖片和遮罩
    image = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')  # 轉換為灰階

    # 確保尺寸一致
    if image.size != mask.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)

    # 將圖片轉換為 RGBA 以支援透明度
    image_rgba = image.convert('RGBA')

    # 將遮罩轉換為 numpy 陣列
    mask_array = np.array(mask)

    # 創建 alpha 通道（遮罩的白色部分會變成不透明）
    alpha = Image.fromarray(mask_array)

    # 將 alpha 通道應用到圖片
    image_rgba.putalpha(alpha)

    # 根據背景類型處理
    if background == 'transparent':
        # 直接保存為 PNG 保持透明度
        image_rgba.save(output_path, 'PNG')
    else:
        # 創建指定顏色的背景
        bg_color = (255, 255, 255) if background == 'white' else (0, 0, 0)
        bg = Image.new('RGB', image.size, bg_color)
        # 將分割後的圖片貼到背景上
        bg.paste(image_rgba, mask=alpha)
        bg.save(output_path, 'PNG')


def process_directory(image_dir, mask_dir, output_dir, background='transparent'):
    """
    批次處理整個目錄的圖片

    Args:
        image_dir: 原始圖片目錄
        mask_dir: 遮罩目錄
        output_dir: 輸出目錄
        background: 背景類型
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    # 建立輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)

    # 取得所有圖片檔案
    image_files = sorted(list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpg')))

    if not image_files:
        print(f"❌ 在 {image_dir} 中沒有找到圖片")
        return

    print(f"📁 原始圖片目錄: {image_dir}")
    print(f"🎭 遮罩目錄: {mask_dir}")
    print(f"💾 輸出目錄: {output_dir}")
    print(f"🖼️  找到 {len(image_files)} 張圖片\n")

    # 處理每張圖片
    success_count = 0
    error_count = 0

    for image_file in tqdm(image_files, desc="處理圖片"):
        # 找到對應的遮罩檔案
        mask_file = mask_dir / image_file.name

        if not mask_file.exists():
            print(f"⚠️  找不到遮罩: {mask_file}")
            error_count += 1
            continue

        # 輸出檔案路徑
        output_file = output_dir / image_file.name

        try:
            apply_mask_to_image(image_file, mask_file, output_file, background)
            success_count += 1
        except Exception as e:
            print(f"❌ 處理 {image_file.name} 時發生錯誤: {e}")
            error_count += 1

    print(f"\n✅ 完成！成功處理 {success_count} 張圖片")
    if error_count > 0:
        print(f"❌ {error_count} 張圖片處理失敗")
    print(f"📂 結果已儲存至: {output_dir}")


def main():
    if len(sys.argv) < 3:
        print("使用方式: python apply_mask.py <圖片目錄> <遮罩目錄> [輸出目錄] [背景類型]")
        print("\n參數說明:")
        print("  圖片目錄: 原始圖片所在目錄")
        print("  遮罩目錄: 遮罩圖片所在目錄")
        print("  輸出目錄: (可選) 輸出目錄，預設為圖片目錄_segmented")
        print("  背景類型: (可選) transparent/white/black，預設為 transparent")
        print("\n範例:")
        print("  python apply_mask.py datasets/mvtec_ad/hazelnut/train/good datasets/mvtec_ad/hazelnut/train/good_mask")
        print("  python apply_mask.py datasets/mvtec_ad/hazelnut/train/good datasets/mvtec_ad/hazelnut/train/good_mask datasets/mvtec_ad/hazelnut/train/good_segmented white")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    mask_dir = Path(sys.argv[2])

    # 檢查目錄是否存在
    if not image_dir.exists():
        print(f"❌ 圖片目錄不存在: {image_dir}")
        sys.exit(1)

    if not mask_dir.exists():
        print(f"❌ 遮罩目錄不存在: {mask_dir}")
        sys.exit(1)

    # 設定輸出目錄
    if len(sys.argv) >= 4:
        output_dir = Path(sys.argv[3])
    else:
        output_dir = image_dir.parent / f"{image_dir.name}_segmented"

    # 設定背景類型
    background = 'transparent'
    if len(sys.argv) >= 5:
        background = sys.argv[4]
        if background not in ['transparent', 'white', 'black']:
            print(f"⚠️  無效的背景類型: {background}，使用預設值 'transparent'")
            background = 'transparent'

    # 處理圖片
    process_directory(image_dir, mask_dir, output_dir, background)


if __name__ == "__main__":
    main()
