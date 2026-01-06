#!/usr/bin/env python3
"""
為圖片添加高斯雜訊
"""
import argparse
import numpy as np
from PIL import Image
import os


def add_gaussian_noise(image_path, output_path, noise_ratio=0.1, seed=None, output_size=None):
    """
    為圖片添加高斯雜訊

    Args:
        image_path: 輸入圖片路徑
        output_path: 輸出圖片路徑
        noise_ratio: 雜訊比例 (0.0-1.0)，相對於圖片像素值範圍 [0, 255]
        seed: 隨機種子，用於可重現的結果
        output_size: 輸出圖片大小 (width, height)，例如 (512, 512)
    """
    if seed is not None:
        np.random.seed(seed)

    # 讀取圖片
    image = Image.open(image_path)
    image_array = np.array(image, dtype=np.float32)

    # 生成高斯雜訊
    # 標準差 = noise_ratio * 255（假設圖片範圍是 0-255）
    noise_std = noise_ratio * 255.0
    noise = np.random.normal(0, noise_std, image_array.shape)

    # 添加雜訊
    noisy_image = image_array + noise

    # 裁剪到 [0, 255] 範圍
    noisy_image = np.clip(noisy_image, 0, 255)

    # 轉換回 uint8
    noisy_image = noisy_image.astype(np.uint8)

    # 轉換為 PIL 圖片
    output_image = Image.fromarray(noisy_image)

    # 調整大小（如果指定）
    if output_size is not None:
        output_image = output_image.resize(output_size, Image.LANCZOS)
        print(f"圖片已調整為: {output_size[0]}x{output_size[1]}")

    # 保存圖片
    output_image.save(output_path)

    print(f"已添加高斯雜訊 (比例: {noise_ratio})")
    print(f"輸入: {image_path}")
    print(f"輸出: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='為圖片添加高斯雜訊')
    parser.add_argument('input', type=str, help='輸入圖片路徑')
    parser.add_argument('output', type=str, help='輸出圖片路徑')
    parser.add_argument('--noise-ratio', type=float, default=0.1,
                        help='雜訊比例 (0.0-1.0)，預設 0.1。數值越大雜訊越強')
    parser.add_argument('--seed', type=int, default=None,
                        help='隨機種子，用於可重現的結果')
    parser.add_argument('--size', type=int, nargs=2, default=None,
                        help='輸出圖片大小 (寬 高)，例如 512 512')

    args = parser.parse_args()

    # 檢查輸入文件是否存在
    if not os.path.exists(args.input):
        print(f"錯誤: 輸入文件不存在: {args.input}")
        return

    # 檢查雜訊比例範圍
    if args.noise_ratio < 0 or args.noise_ratio > 1:
        print(f"警告: 雜訊比例建議在 0.0-1.0 之間，目前為 {args.noise_ratio}")

    # 創建輸出目錄（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 處理輸出大小
    output_size = tuple(args.size) if args.size else None

    # 添加雜訊
    add_gaussian_noise(args.input, args.output, args.noise_ratio, args.seed, output_size)


if __name__ == '__main__':
    main()

# .env/bin/python add_gaussian_noise.py '/home/nknul40s/bluestar-research/TengTengDiff/datasets/mvtec_ad/hazelnut/test/hole/000.png' output.png --size 512 512  --noise-ratio 0