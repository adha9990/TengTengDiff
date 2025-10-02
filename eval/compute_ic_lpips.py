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
                    lpips_net: str = 'vgg', batch_size: int = 32,
                    max_images: int = None, n_clusters: int = 10,
                    images_per_cluster: int = 50, use_clustering: bool = True) -> Dict[str, float]:
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

    if use_clustering:
        # Cluster-based IC-LPIPS (Ojha et al. 2021)
        required_images = n_clusters * images_per_cluster
        if n_images < required_images:
            print(f"警告: 圖片數量 ({n_images}) 少於所需 ({required_images}), 調整為 {n_images // n_clusters} 張/cluster")
            images_per_cluster = max(2, n_images // n_clusters)

        cluster_results = []
        all_distances = []

        for cluster_id in range(n_clusters):
            # 隨機採樣該 cluster 的圖片
            if cluster_id * images_per_cluster + images_per_cluster <= n_images:
                # 順序分配（也可以使用隨機採樣）
                start_idx = cluster_id * images_per_cluster
                end_idx = start_idx + images_per_cluster
                cluster_images = images[start_idx:end_idx]
            else:
                # 最後一個 cluster 可能包含剩餘的所有圖片
                indices = np.random.choice(n_images, min(images_per_cluster, n_images), replace=False)
                cluster_images = images[indices]

            if len(cluster_images) < 2:
                continue

            # 計算該 cluster 內的 pairwise LPIPS
            cluster_distances = compute_pairwise_lpips(cluster_images, lpips_model, batch_size)

            cluster_mean = np.mean(cluster_distances)
            cluster_std = np.std(cluster_distances)

            cluster_results.append({
                'cluster_id': cluster_id,
                'mean': cluster_mean,
                'std': cluster_std,
                'n_images': len(cluster_images),
                'n_pairs': len(cluster_distances)
            })

            all_distances.extend(cluster_distances.tolist())
            print(f"  Cluster {cluster_id}: {cluster_mean:.4f} ± {cluster_std:.4f} ({len(cluster_images)} 張圖片)")

        # 計算跨 cluster 的統計量
        cluster_means = [r['mean'] for r in cluster_results]
        overall_mean = np.mean(cluster_means)
        overall_std = np.std(cluster_means)

        return {
            'mean': overall_mean,
            'std': overall_std,
            'min': np.min(all_distances),
            'max': np.max(all_distances),
            'median': np.median(all_distances),
            'n_images': n_images,
            'n_clusters': len(cluster_results),
            'cluster_results': cluster_results,
            'all_pairs': len(all_distances)
        }
    else:
        # 原始方法：計算所有圖片對的 LPIPS
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
    parser.add_argument('--lpips_net', type=str, default='vgg',
                        choices=['alex', 'vgg', 'squeeze'],
                        help='LPIPS network to use (default: vgg, matching Ojha et al. 2021)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to use (default: all)')
    parser.add_argument('--n_clusters', type=int, default=10,
                        help='Number of clusters for IC-LPIPS (default: 10)')
    parser.add_argument('--images_per_cluster', type=int, default=50,
                        help='Images per cluster (default: 50)')
    parser.add_argument('--no_clustering', action='store_true',
                        help='Disable clustering, compute all pairwise distances')
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
        max_images=args.max_images,
        n_clusters=args.n_clusters,
        images_per_cluster=args.images_per_cluster,
        use_clustering=not args.no_clustering
    )

    print(f"\n{'='*50}")
    print(f"IC-LPIPS 結果: {args.image_dir}")
    print(f"{'='*50}")
    print(f"平均 LPIPS 距離: {results['mean']:.4f} ± {results['std']:.4f}")
    print(f"中位數 LPIPS 距離: {results['median']:.4f}")
    print(f"最小 LPIPS 距離: {results['min']:.4f}")
    print(f"最大 LPIPS 距離: {results['max']:.4f}")
    print(f"圖片數量: {results['n_images']}")
    if 'n_clusters' in results:
        print(f"Cluster 數量: {results['n_clusters']}")
        print(f"總圖片對數量: {results['all_pairs']}")
    else:
        print(f"圖片對數量: {results['n_pairs']}")
    print(f"{'='*50}")

    output_file = os.path.join(os.path.dirname(args.image_dir.rstrip('/')), 'ic_lpips.txt')
    with open(output_file, 'w') as f:
        f.write(f"Image Directory: {args.image_dir}\n")
        f.write(f"LPIPS Network: {args.lpips_net}\n")
        f.write(f"Number of Images: {results['n_images']}\n")
        if 'n_clusters' in results:
            f.write(f"Clustering: Enabled\n")
            f.write(f"Number of Clusters: {results['n_clusters']}\n")
            f.write(f"Images per Cluster: {args.images_per_cluster}\n")
            f.write(f"Total Pairs: {results['all_pairs']}\n")
        else:
            f.write(f"Clustering: Disabled\n")
            f.write(f"Number of Pairs: {results['n_pairs']}\n")
        f.write(f"Mean LPIPS Distance: {results['mean']:.4f} ± {results['std']:.4f}\n")
        f.write(f"Median LPIPS Distance: {results['median']:.4f}\n")
        f.write(f"Min LPIPS Distance: {results['min']:.4f}\n")
        f.write(f"Max LPIPS Distance: {results['max']:.4f}\n")

        if 'cluster_results' in results:
            f.write(f"\n--- Per-Cluster Results ---\n")
            for cluster in results['cluster_results']:
                f.write(f"Cluster {cluster['cluster_id']}: {cluster['mean']:.4f} ± {cluster['std']:.4f} ")
                f.write(f"({cluster['n_images']} images, {cluster['n_pairs']} pairs)\n")

    print(f"\n結果已儲存至: {output_file}")


if __name__ == '__main__':
    main()