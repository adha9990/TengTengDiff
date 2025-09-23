import argparse
import random
import torch
import torch.nn as nn
from torchvision import utils
from tqdm import tqdm
import sys
import lpips
from torchvision import transforms, utils
from torch.utils import data
import os
from PIL import Image
import numpy as np

lpips_fn = lpips.LPIPS(net='vgg').cuda()
preprocess = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def compute_diversity_only(tar_path):
    """只計算生成圖片之間的多樣性（不需要參考圖片）"""
    if not os.path.exists(tar_path):
        return None
    
    files_list = os.listdir(tar_path)
    if len(files_list) < 2:
        print(f"Not enough images for diversity computation: {len(files_list)}")
        return None
    
    # 限制最多使用50張圖片以加快計算
    if len(files_list) > 50:
        import random
        random.shuffle(files_list)
        files_list = files_list[:50]
    
    dists = []
    with torch.no_grad():
        for i in range(len(files_list)):
            for j in range(i + 1, len(files_list)):
                img1_path = os.path.join(tar_path, files_list[i])
                img2_path = os.path.join(tar_path, files_list[j])
                
                try:
                    img1 = Image.open(img1_path).convert('RGB')
                    img2 = Image.open(img2_path).convert('RGB')
                    
                    tensor1 = preprocess(img1).unsqueeze(0).to(device)
                    tensor2 = preprocess(img2).unsqueeze(0).to(device)
                    
                    dist = lpips_fn(tensor1, tensor2)
                    dists.append(dist.item())
                except Exception as e:
                    print(f"Error processing {img1_path} or {img2_path}: {e}")
                    continue
    
    if len(dists) > 0:
        return np.mean(dists)
    return None

def ic_lpips(sample_name, anomaly_name, generate_data_path, mvtec_path):
    print(sample_name,anomaly_name)
    tar_path='%s/%s/%s/image'%(generate_data_path, sample_name, anomaly_name)

    ori_path='%s/%s/test/%s'%(mvtec_path, sample_name, anomaly_name)
    
    # 檢查路徑是否存在
    if not os.path.exists(tar_path):
        print(f"Warning: Generated data path not found: {tar_path}")
        return None
    
    if not os.path.exists(ori_path):
        # 如果找不到對應的異常類型，使用訓練資料中的 good 圖片
        ori_path = '%s/%s/train/good'%(mvtec_path, sample_name)
        if not os.path.exists(ori_path):
            # 如果還是找不到，使用測試資料中的 good 圖片
            ori_path = '%s/%s/test/good'%(mvtec_path, sample_name)
            if not os.path.exists(ori_path):
                print(f"Warning: Cannot find any reference images for {sample_name}")
                # 只計算生成圖片之間的多樣性
                return compute_diversity_only(tar_path)
    
    with torch.no_grad():
        # 使用 DualAnoDiff 的邏輯：將原始圖片數量除以 3
        l = len(os.listdir(ori_path)) // 3
        avg_dist = torch.zeros([l, ])
        files_list=os.listdir(tar_path)
        input_tensors1=[]
        clusters=[[] for i in range(l)]
        for k in range(l):
            input1_path = os.path.join(ori_path, '%03d.png' % k)
            if not os.path.exists(input1_path):
                print(f"Warning: Reference image not found: {input1_path}")
                continue
            input_image1 = Image.open(input1_path).convert('RGB')
            input_tensor1 = preprocess(input_image1)
            input_tensor1 = input_tensor1.to(device)
            input_tensors1.append(input_tensor1)
        # 如果沒有參考圖片，則返回 None
        if len(input_tensors1) == 0:
            print(f"No valid reference images found for {sample_name}/{anomaly_name}")
            return None
            
        for i in range(len(files_list)):
            min_dist = 999999999
            max_ind = 0
            input2_path = os.path.join(tar_path, files_list[i])
            input_image2 = Image.open(input2_path).convert('RGB')
            input_tensor2 = preprocess(input_image2)
            input_tensor2 = input_tensor2.to(device)
            for k in range(len(input_tensors1)):
                dist = lpips_fn(input_tensors1[k], input_tensor2)
                if dist <= min_dist:
                    max_ind = k
                    min_dist = dist
            clusters[max_ind].append(input2_path)
        cluster_size=50
        for k in range(len(input_tensors1)):  # 使用實際的參考圖片數量
            print(k)
            files_list=clusters[k]
            if len(files_list) == 0:
                continue
            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            dists = []
            for i in range(len(files_list)):
                for j in range(i + 1, len(files_list)):
                    input1_path = files_list[i]
                    input2_path = files_list[j]

                    input_image1 = Image.open(input1_path).convert('RGB')
                    input_image2 = Image.open(input2_path).convert('RGB')

                    input_tensor1 = preprocess(input_image1)
                    input_tensor2 = preprocess(input_image2)

                    input_tensor1 = input_tensor1.to(device)
                    input_tensor2 = input_tensor2.to(device)

                    dist = lpips_fn(input_tensor1, input_tensor2)

                    dists.append(dist)
            if len(dists) > 0:
                dists = torch.tensor(dists)
                avg_dist[k] = dists.mean()
        valid_dists = avg_dist[~torch.isnan(avg_dist)]
        if valid_dists.numel() > 0:
            return valid_dists.mean()
        else:
            return None


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_data_path', type=str, default='generate_data', 
                        help='Path to generated data directory')
    parser.add_argument('--mvtec_path', type=str, default='datasets/mvtec_ad',
                        help='Path to MVTec dataset')
    parser.add_argument('--output_file', type=str, default='eval_result/ic_lpips_results.csv',
                        help='Output CSV file name')
    parser.add_argument('--sample_name', type=str, default='all',
                        help='Specific sample to evaluate, or "all" for all samples')
    
    args = parser.parse_args()
    
    sample_names=[
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
    import csv
    
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
    
    # 開始評估
    with open(args.output_file, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_name', 'anomaly_type', 'ic_lpips_score'])
    
    for sample_name in available_samples:
        dis=0
        cnt=0
        sample_path = os.path.join(args.generate_data_path, sample_name)
        for anomaly_name in os.listdir(sample_path):
            result = ic_lpips(sample_name, anomaly_name, args.generate_data_path, args.mvtec_path)
            if result is not None:
                dis+=result
                cnt+=1
                with open(args.output_file, "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([sample_name, anomaly_name, str(float(result))])
        
        if cnt > 0:
            avg_score = float(dis/cnt)
            with open(args.output_file, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([sample_name, 'average', str(avg_score)])
            print(f"{sample_name} average IC-LPIPS: {avg_score}")
            
