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

def ic_lpips(sample_name,anomaly_name, base_generate_path, base_mvtec_path):
    print(sample_name,anomaly_name)
    tar_path='%s/%s/%s/image'%(base_generate_path, sample_name,anomaly_name)

    ori_path='%s/%s/test/%s'%(base_mvtec_path, sample_name,anomaly_name)
    with torch.no_grad():
        l = len(os.listdir(ori_path)) // 3
        avg_dist = torch.zeros([l, ])
        files_list=os.listdir(tar_path)
        input_tensors1=[]
        clusters=[[] for i in range(l)]
        for k in range(l):
            input1_path = os.path.join(ori_path, '%03d.png' % k)
            input_image1 = Image.open(input1_path).convert('RGB')
            input_tensor1 = preprocess(input_image1)
            input_tensor1 = input_tensor1.to(device)
            input_tensors1.append(input_tensor1)
        for i in range(len(files_list)):
            min_dist = 999999999
            input2_path = os.path.join(tar_path, files_list[i])
            input_image2 = Image.open(input2_path).convert('RGB')
            input_tensor2 = preprocess(input_image2)
            input_tensor2 = input_tensor2.to(device)
            for k in range(l):
                dist = lpips_fn(input_tensors1[k], input_tensor2)
                if dist <= min_dist:
                    max_ind = k
                    min_dist = dist
            clusters[max_ind].append(input2_path)
        cluster_size=50
        for k in range(l):
            print(k)
            files_list=clusters[k]
            random.shuffle(files_list)
            files_list = files_list[:cluster_size]
            dists = []
            for i in range(len(files_list)):
                for j in range(i + 1, len(files_list)):
                    input1_path = files_list[i]
                    input2_path = files_list[j]

                    input_image1 = Image.open(input1_path)
                    input_image2 = Image.open(input2_path)

                    input_tensor1 = preprocess(input_image1)
                    input_tensor2 = preprocess(input_image2)

                    input_tensor1 = input_tensor1.to(device)
                    input_tensor2 = input_tensor2.to(device)

                    dist = lpips_fn(input_tensor1, input_tensor2)

                    dists.append(dist)
            dists = torch.tensor(dists)
            avg_dist[k] = dists.mean()
        return avg_dist[~torch.isnan(avg_dist)].mean()









if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_data_path', type=str,
                       default='/home/bluestar/research/TengTengDiff/generate_data',
                       help='Path to generated data')
    parser.add_argument('--mvtec_path', type=str,
                       default='/home/bluestar/research/TengTengDiff/datasets/mvtec_ad',
                       help='Path to MVTec dataset')
    parser.add_argument('--sample_name', type=str, default='hazelnut',
                       help='Sample name to evaluate (default: hazelnut)')
    parser.add_argument('--output', type=str, default='ic_lpips_results.csv',
                       help='Output CSV file name')
    parser.add_argument('--direct_path_mode', action='store_true',
                       help='Enable direct path mode for single image directory evaluation')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Direct path to image directory (used with --direct_path_mode)')
    parser.add_argument('--anomaly_name', type=str, default=None,
                       help='Anomaly name (used with --direct_path_mode)')
    args = parser.parse_args()

    import csv

    if args.direct_path_mode:
        # 直接路徑模式：評估單一圖片目錄
        if not args.image_path or not args.anomaly_name:
            print("錯誤: 直接路徑模式需要 --image_path 和 --anomaly_name 參數")
            sys.exit(1)

        if not os.path.exists(args.image_path):
            print(f"錯誤: 圖片路徑不存在: {args.image_path}")
            sys.exit(1)

        # 準備 CSV 文件
        with open(args.output, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Sample Name', 'Anomaly Name', 'IC-LPIPS Score'])

        # 調用 ic_lpips 函數，但需要適配路徑
        # 原函數期望: base_generate_path/{sample_name}/{anomaly_name}/image
        # 我們需要傳遞父目錄作為 base_generate_path
        print(f"評估: {args.sample_name} - {args.anomaly_name}")

        # 檢查 MVTec 異常類型目錄
        # anomaly_name 可能是 "stage1-print-dual/checkpoint-1000" 或 "stage2-hole-dual/checkpoint-5000"
        # 我們需要從中提取實際的異常類型（如 "print", "hole"）

        # 先從 MVTec 目錄中獲取所有可用的異常類型
        mvtec_test_dir = os.path.join(args.mvtec_path, args.sample_name, 'test')
        if not os.path.exists(mvtec_test_dir):
            print(f"錯誤: MVTec 測試目錄不存在: {mvtec_test_dir}")
            sys.exit(1)

        # 獲取所有異常類型（排除 'good'）
        available_anomalies = [d for d in os.listdir(mvtec_test_dir)
                              if os.path.isdir(os.path.join(mvtec_test_dir, d)) and d != 'good']

        if not available_anomalies:
            print(f"錯誤: 在 {mvtec_test_dir} 中找不到任何異常類型")
            sys.exit(1)

        # 在 anomaly_name 中搜尋匹配的異常類型
        anomaly_type = None
        for anomaly in available_anomalies:
            if anomaly.lower() in args.anomaly_name.lower():
                anomaly_type = anomaly
                print(f"從路徑 '{args.anomaly_name}' 推斷出異常類型: {anomaly_type}")
                break

        # 如果沒有找到匹配的異常類型，報錯而非默默使用第一個
        if not anomaly_type:
            print(f"錯誤: 無法從路徑 '{args.anomaly_name}' 推斷異常類型")
            print(f"可用的異常類型: {', '.join(available_anomalies)}")
            print(f"請確保路徑中包含其中一個異常類型名稱")
            sys.exit(1)

        # 檢查 MVTec 測試目錄
        mvtec_anomaly_dir = os.path.join(args.mvtec_path, args.sample_name, 'test', anomaly_type)
        if not os.path.exists(mvtec_anomaly_dir):
            print(f"錯誤: MVTec 異常目錄不存在: {mvtec_anomaly_dir}")
            sys.exit(1)

        # 修改 ic_lpips 函數以支持直接路徑
        tar_path = args.image_path
        ori_path = mvtec_anomaly_dir

        with torch.no_grad():
            l = len(os.listdir(ori_path)) // 3
            avg_dist = torch.zeros([l, ])
            files_list=os.listdir(tar_path)
            input_tensors1=[]
            clusters=[[] for i in range(l)]

            for k in range(l):
                input1_path = os.path.join(ori_path, '%03d.png' % k)
                input_image1 = Image.open(input1_path).convert('RGB')
                input_tensor1 = preprocess(input_image1)
                input_tensor1 = input_tensor1.to(device)
                input_tensors1.append(input_tensor1)

            for i in range(len(files_list)):
                min_dist = 999999999
                input2_path = os.path.join(tar_path, files_list[i])
                input_image2 = Image.open(input2_path).convert('RGB')
                input_tensor2 = preprocess(input_image2)
                input_tensor2 = input_tensor2.to(device)
                for k in range(l):
                    dist = lpips_fn(input_tensors1[k], input_tensor2)
                    if dist <= min_dist:
                        max_ind = k
                        min_dist = dist
                clusters[max_ind].append(input2_path)

            cluster_size=50
            for k in range(l):
                print(f"處理中 {k}/{l}")
                files_list=clusters[k]
                random.shuffle(files_list)
                files_list = files_list[:cluster_size]
                dists = []
                for i in range(len(files_list)):
                    for j in range(i + 1, len(files_list)):
                        input1_path = files_list[i]
                        input2_path = files_list[j]

                        input_image1 = Image.open(input1_path)
                        input_image2 = Image.open(input2_path)

                        input_tensor1 = preprocess(input_image1)
                        input_tensor2 = preprocess(input_image2)

                        input_tensor1 = input_tensor1.to(device)
                        input_tensor2 = input_tensor2.to(device)

                        dist = lpips_fn(input_tensor1, input_tensor2)

                        dists.append(dist)
                dists = torch.tensor(dists)
                avg_dist[k] = dists.mean()

            score = avg_dist[~torch.isnan(avg_dist)].mean()

        print(f"\n{args.sample_name} - {args.anomaly_name}: {score:.4f}")
        with open(args.output, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([args.sample_name, args.anomaly_name, f"{score:.4f}"])

        print(f"\nResults saved to {args.output}")
    else:
        # 原有的批量評估模式
        sample_names=[args.sample_name] if args.sample_name != 'all' else [
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

        with open(args.output, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Sample Name', 'IC-LPIPS Score'])

        for sample_name in sample_names:
            sample_path = os.path.join(args.generate_data_path, sample_name)
            if not os.path.exists(sample_path):
                print(f"Skipping {sample_name}: path not found")
                continue

            dis=0
            cnt=0
            for anomaly_name in os.listdir(sample_path):
                anomaly_path = os.path.join(sample_path, anomaly_name, 'image')
                if not os.path.exists(anomaly_path):
                    continue
                dis+=ic_lpips(sample_name, anomaly_name, args.generate_data_path, args.mvtec_path)
                cnt+=1

            if cnt > 0:
                avg_score = float(dis/cnt)
                print(f"\n{sample_name}: {avg_score:.4f}")
                with open(args.output, "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([sample_name, f"{avg_score:.4f}"])

        print(f"\nResults saved to {args.output}")
            
