#!/usr/bin/env python
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from typing import List, Tuple
from tqdm import tqdm


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

        return img


def get_inception_model(device: torch.device) -> nn.Module:
    model = models.inception_v3(pretrained=True, transform_input=False)
    model = model.to(device)
    model.eval()
    return model


def get_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device, num_classes: int = 1000) -> np.ndarray:
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="提取特徵"):
            batch = batch.to(device)
            logits = model(batch)

            if isinstance(logits, tuple):
                logits = logits[0]

            probs = F.softmax(logits, dim=1).cpu().numpy()
            predictions.append(probs)

    predictions = np.concatenate(predictions, axis=0)
    return predictions


def compute_inception_score(predictions: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    scores = []
    n = predictions.shape[0]
    split_size = n // splits

    for i in range(splits):
        start = i * split_size
        end = (i + 1) * split_size if i < splits - 1 else n
        part = predictions[start:end]

        p_yx = part
        p_y = np.mean(part, axis=0, keepdims=True)

        kl = part * (np.log(part + 1e-8) - np.log(p_y + 1e-8))
        kl = np.sum(kl, axis=1)
        kl = np.mean(kl)

        scores.append(np.exp(kl))

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    return mean_score, std_score


def main():
    parser = argparse.ArgumentParser(description='Compute Inception Score for generated images')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing generated images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing (default: 32)')
    parser.add_argument('--splits', type=int, default=10,
                        help='Number of splits for IS calculation (default: 10)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available)')

    args = parser.parse_args()

    if not os.path.exists(args.image_dir):
        raise ValueError(f"圖片目錄不存在: {args.image_dir}")

    device = torch.device(args.device)
    print(f"使用裝置: {device}")

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(args.image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    print("載入 InceptionV3 模型...")
    model = get_inception_model(device)

    print("計算預測...")
    predictions = get_predictions(model, dataloader, device)

    print(f"使用 {args.splits} 個分割計算 Inception Score...")
    mean_score, std_score = compute_inception_score(predictions, splits=args.splits)

    print(f"\n{'='*50}")
    print(f"結果: {args.image_dir}")
    print(f"{'='*50}")
    print(f"Inception Score: {mean_score:.4f} ± {std_score:.4f}")
    print(f"圖片數量: {len(dataset)}")
    print(f"{'='*50}")

    output_file = os.path.join(os.path.dirname(args.image_dir.rstrip('/')), 'inception_score.txt')
    with open(output_file, 'w') as f:
        f.write(f"Image Directory: {args.image_dir}\n")
        f.write(f"Number of Images: {len(dataset)}\n")
        f.write(f"Inception Score: {mean_score:.4f} ± {std_score:.4f}\n")
        f.write(f"Splits: {args.splits}\n")

    print(f"\n結果已儲存至: {output_file}")


if __name__ == '__main__':
    main()