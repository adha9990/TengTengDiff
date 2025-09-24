#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple, Dict
import itertools
from tqdm import tqdm
import lpips
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = []

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.images.append(os.path.join(root, file))

        if len(self.images) == 0:
            raise ValueError(f"在 {image_dir} 中找不到圖片")

        print(f"找到 {len(self.images)} 張圖片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, img_path


def load_images(image_dir: str, device: torch.device, size: int = 256) -> Tuple[torch.Tensor, List[str]]:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = ImageDataset(image_dir, transform=transform)

    images = []
    paths = []

    for img, path in dataset:
        images.append(img)
        paths.append(path)

    if len(images) == 0:
        raise ValueError(f"在 {image_dir} 中找不到有效的圖片")

    images = torch.stack(images).to(device)
    return images, paths


def compute_pairwise_lpips(images: torch.Tensor, lpips_model: nn.Module,
                          batch_size: int = 32) -> np.ndarray:
    n = images.shape[0]
    distances = []

    pairs = list(itertools.combinations(range(n), 2))
    total_pairs = len(pairs)

    print(f"正在計算 {total_pairs} 對圖片的 LPIPS...")

    with torch.no_grad():
        for i in tqdm(range(0, total_pairs, batch_size), desc="處理圖片對"):
            batch_pairs = pairs[i:i + batch_size]

            batch_img1 = torch.stack([images[p[0]] for p in batch_pairs])
            batch_img2 = torch.stack([images[p[1]] for p in batch_pairs])

            dist = lpips_model(batch_img1, batch_img2)
            distances.extend(dist.squeeze().cpu().numpy().tolist())

    return np.array(distances)


def compute_ic_lpips(image_dir: str, device: torch.device,
                    lpips_net: str = 'alex', batch_size: int = 32,
                    max_images: int = None) -> Dict[str, float]:
    print(f"載入 LPIPS 模型 ({lpips_net})...")
    lpips_model = lpips.LPIPS(net=lpips_net).to(device)
    lpips_model.eval()

    print(f"從 {image_dir} 載入圖片...")
    images, paths = load_images(image_dir, device)

    if max_images and len(images) > max_images:
        print(f"從總共 {len(images)} 張圖片中抽樣 {max_images} 張")
        indices = np.random.choice(len(images), max_images, replace=False)
        images = images[indices]
        paths = [paths[i] for i in indices]

    n_images = len(images)
    print(f"為 {n_images} 張圖片計算 IC-LPIPS...")

    if n_images < 2:
        raise ValueError(f"需要至少 2 張圖片來計算 IC-LPIPS，但只有 {n_images} 張")

    distances = compute_pairwise_lpips(images, lpips_model, batch_size)

    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    median_distance = np.median(distances)

    return {
        'mean': mean_distance,
        'std': std_distance,
        'min': min_distance,
        'max': max_distance,
        'median': median_distance,
        'n_images': n_images,
        'n_pairs': len(distances)
    }


def main():
    parser = argparse.ArgumentParser(description='Compute IC-LPIPS for generated images')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing generated images')
    parser.add_argument('--lpips_net', type=str, default='alex',
                        choices=['alex', 'vgg', 'squeeze'],
                        help='LPIPS network to use (default: alex)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to use (default: all)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')

    args = parser.parse_args()

    if not os.path.exists(args.image_dir):
        raise ValueError(f"圖片目錄不存在: {args.image_dir}")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"使用裝置: {device}")

    results = compute_ic_lpips(
        args.image_dir,
        device,
        lpips_net=args.lpips_net,
        batch_size=args.batch_size,
        max_images=args.max_images
    )

    print(f"\n{'='*50}")
    print(f"IC-LPIPS 結果: {args.image_dir}")
    print(f"{'='*50}")
    print(f"平均 LPIPS 距離: {results['mean']:.4f} ± {results['std']:.4f}")
    print(f"中位數 LPIPS 距離: {results['median']:.4f}")
    print(f"最小 LPIPS 距離: {results['min']:.4f}")
    print(f"最大 LPIPS 距離: {results['max']:.4f}")
    print(f"圖片數量: {results['n_images']}")
    print(f"圖片對數量: {results['n_pairs']}")
    print(f"{'='*50}")

    output_file = os.path.join(os.path.dirname(args.image_dir.rstrip('/')), 'ic_lpips.txt')
    with open(output_file, 'w') as f:
        f.write(f"Image Directory: {args.image_dir}\n")
        f.write(f"LPIPS Network: {args.lpips_net}\n")
        f.write(f"Number of Images: {results['n_images']}\n")
        f.write(f"Number of Pairs: {results['n_pairs']}\n")
        f.write(f"Mean LPIPS Distance: {results['mean']:.4f} ± {results['std']:.4f}\n")
        f.write(f"Median LPIPS Distance: {results['median']:.4f}\n")
        f.write(f"Min LPIPS Distance: {results['min']:.4f}\n")
        f.write(f"Max LPIPS Distance: {results['max']:.4f}\n")

    print(f"\n結果已儲存至: {output_file}")


if __name__ == '__main__':
    main()