import os
import argparse
import sys
import torch
from torch_fidelity import calculate_metrics
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--generate_data_path', type=str, default='generate_data',
                    help='Path to generated data directory')
parser.add_argument('--output_file', type=str, default='eval_result/inception_score_results.csv',
                    help='Output CSV file name')
parser.add_argument('--sample_name', type=str, default='all',
                    help='Specific sample to evaluate, or "all" for all samples')
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU ID to use')
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch size for evaluation')

args = parser.parse_args()

# 設定 GPU
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    device = f'cuda:{args.gpu}'
else:
    device = 'cpu'

sample_names = [
    'capsule',
    'bottle',
    'carpet',
    'leather',
    'pill',
    'transistor',
    'tile',
    'cable',
    'zipper',
    'toothbrush',
    'metal_nut',
    'hazelnut',
    'screw',
    'grid',
    'wood'
]

# 根據參數決定要評估哪些樣本
if args.sample_name != 'all':
    sample_names = [args.sample_name]

# 檢查哪些樣本有生成資料
available_samples = []
for sample_name in sample_names:
    sample_path = os.path.join(args.generate_data_path, sample_name)
    if os.path.exists(sample_path):
        available_samples.append(sample_name)

if not available_samples:
    print(f"Error: No generated data found in {args.generate_data_path}")
    sys.exit(1)

print(f"Found generated data for: {available_samples}")

with open(args.output_file, "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['sample_name', 'anomaly_type', 'inception_score_mean', 'inception_score_std'])

for sample_name in available_samples:
    dir_name = os.path.join(args.generate_data_path, sample_name)
    sample_scores = []
    
    for anomaly_name in os.listdir(dir_name):
        print(f"\nEvaluating {sample_name}/{anomaly_name}")
        image_path = os.path.join(dir_name, anomaly_name, 'image')
        
        if not os.path.exists(image_path):
            print(f"Warning: No image directory found at {image_path}")
            continue
        
        # 檢查圖片數量
        images = [f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) < 10:
            print(f"Warning: Only {len(images)} images found. Inception Score requires at least 10 images.")
            if len(images) < 2:
                continue
        
        print(f"Computing Inception Score for {len(images)} images...")
        
        try:
            # 使用 torch-fidelity 計算 Inception Score
            metrics = calculate_metrics(
                input1=image_path,
                cuda=(device != 'cpu'),
                isc=True,
                batch_size=args.batch_size,
                verbose=False
            )
            
            is_mean = metrics['inception_score_mean']
            is_std = metrics['inception_score_std']
            
            print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
            
            with open(args.output_file, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([sample_name, anomaly_name, f'{is_mean:.4f}', f'{is_std:.4f}'])
            
            sample_scores.append(is_mean)
            
        except Exception as e:
            print(f"Error computing IS for {sample_name}/{anomaly_name}: {e}")
            continue
    
    # 計算樣本平均
    if sample_scores:
        import numpy as np
        avg_score = np.mean(sample_scores)
        std_score = np.std(sample_scores)
        with open(args.output_file, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([sample_name, 'average', f'{avg_score:.4f}', f'{std_score:.4f}'])
        print(f"\n{sample_name} average IS: {avg_score:.4f} ± {std_score:.4f}")

print(f"\nResults saved to: {args.output_file}")