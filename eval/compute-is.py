import os
import argparse
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_data_path', type=str,
                       default='/home/bluestar/research/TengTengDiff/generate_data',
                       help='Path to generated data')
    parser.add_argument('--sample_name', type=str, default='hazelnut',
                       help='Sample name to evaluate (default: hazelnut)')
    parser.add_argument('--output', type=str, default='IS_results.csv',
                       help='Output CSV file name')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--direct_path_mode', action='store_true',
                       help='Enable direct path mode for single image directory evaluation')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Direct path to image directory (used with --direct_path_mode)')
    parser.add_argument('--anomaly_name', type=str, default=None,
                       help='Anomaly name (used with --direct_path_mode)')
    args = parser.parse_args()

    if args.direct_path_mode:
        # 直接路徑模式：評估單一圖片目錄
        if not args.image_path or not args.anomaly_name:
            print("錯誤: 直接路徑模式需要 --image_path 和 --anomaly_name 參數")
            import sys
            sys.exit(1)

        if not os.path.exists(args.image_path):
            print(f"錯誤: 圖片路徑不存在: {args.image_path}")
            import sys
            sys.exit(1)

        # 準備 CSV 文件
        with open(args.output, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Sample Name', 'Anomaly Name', 'IS Score'])

        print(f"評估: {args.sample_name} - {args.anomaly_name}")

        # 使用 fidelity 計算 Inception Score
        # 使用虛擬環境中的 fidelity 命令
        fidelity_cmd = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env', 'bin', 'fidelity')
        if not os.path.exists(fidelity_cmd):
            fidelity_cmd = 'fidelity'  # 回退到系統路徑
        os_str = '%s --gpu %d --isc --input1 %s' % (fidelity_cmd, args.gpu, args.image_path)
        f = os.popen(os_str, 'r')
        res = f.readlines()[0]  # res接受返回结果
        f.close()
        print(res)
        data = res[res.index(':')+1:-1]
        print(data)
        score = float(data)

        print(f"\n{args.sample_name} - {args.anomaly_name}: {score:.4f}")
        with open(args.output, "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([args.sample_name, args.anomaly_name, f"{score:.4f}"])

        print(f"\nResults saved to {args.output}")
    else:
        # 原有的批量評估模式
        sample_names = [args.sample_name] if args.sample_name != 'all' else [
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
            writer.writerow(['Sample Name', 'IS Score'])

        for sample_name in sample_names:
            dir_name = os.path.join(args.generate_data_path, sample_name)

            if not os.path.exists(dir_name):
                print(f"Skipping {sample_name}: path not found")
                continue

            dis = 0
            cnt = 0
            for anomaly_name in os.listdir(dir_name):
                image_path = os.path.join(dir_name, anomaly_name, 'image')
                if not os.path.exists(image_path):
                    continue

                print(sample_name, anomaly_name)
                # 使用虛擬環境中的 fidelity 命令
                fidelity_cmd = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env', 'bin', 'fidelity')
                if not os.path.exists(fidelity_cmd):
                    fidelity_cmd = 'fidelity'  # 回退到系統路徑
                os_str = '%s --gpu %d --isc --input1 %s' % (fidelity_cmd, args.gpu, image_path)
                f = os.popen(os_str, 'r')
                res = f.readlines()[0]  # res接受返回结果
                f.close()
                print(res)
                data=res[res.index(':')+1:-1]
                print(data)
                dis += float(data)
                cnt += 1

            if cnt > 0:
                avg_score = float(dis / cnt)
                print(f"\n{sample_name}: {avg_score:.4f}")
                with open(args.output, "a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([sample_name, f"{avg_score:.4f}"])

        print(f"\nResults saved to {args.output}")
