#!/usr/bin/env python3
"""
彙總所有評估結果並計算平均值
"""
import os
import csv
import argparse
from pathlib import Path
from collections import defaultdict

def read_csv_value(csv_path, metric_name):
    """讀取 CSV 文件中的指標值"""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if metric_name in row:
                    return float(row[metric_name])
        return None
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def collect_results(generate_dir, anomalies, checkpoints):
    """收集所有評估結果"""
    results = defaultdict(lambda: defaultdict(dict))

    for anomaly in anomalies:
        for checkpoint in checkpoints:
            checkpoint_dir = f"{generate_dir}/stage2-{anomaly}-dual/checkpoint-{checkpoint}"

            # 讀取 IC-LPIPS 結果
            ic_lpips_path = f"{checkpoint_dir}/ic_lpips_results.csv"
            if os.path.exists(ic_lpips_path):
                ic_lpips = read_csv_value(ic_lpips_path, 'IC-LPIPS Score')
                if ic_lpips is not None:
                    results[checkpoint][anomaly]['ic_lpips'] = ic_lpips

            # 讀取 Inception Score 結果
            is_path = f"{checkpoint_dir}/IS_results.csv"
            if os.path.exists(is_path):
                inception_score = read_csv_value(is_path, 'IS Score')
                if inception_score is not None:
                    results[checkpoint][anomaly]['inception_score'] = inception_score

    return results

def calculate_averages(results, anomalies, checkpoints):
    """計算每個 checkpoint 的平均值"""
    averages = {}

    for checkpoint in checkpoints:
        ic_lpips_values = []
        is_values = []

        for anomaly in anomalies:
            if anomaly in results[checkpoint]:
                if 'ic_lpips' in results[checkpoint][anomaly]:
                    ic_lpips_values.append(results[checkpoint][anomaly]['ic_lpips'])
                if 'inception_score' in results[checkpoint][anomaly]:
                    is_values.append(results[checkpoint][anomaly]['inception_score'])

        averages[checkpoint] = {
            'ic_lpips_avg': sum(ic_lpips_values) / len(ic_lpips_values) if ic_lpips_values else None,
            'is_avg': sum(is_values) / len(is_values) if is_values else None,
            'ic_lpips_count': len(ic_lpips_values),
            'is_count': len(is_values)
        }

    return averages

def generate_summary(results, averages, output_path, anomalies, checkpoints):
    """生成彙總報告"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # 寫入表頭
        writer.writerow(['Checkpoint', 'Anomaly Type', 'IC-LPIPS', 'Inception Score'])
        writer.writerow([])  # 空行

        # 寫入每個 checkpoint 的詳細結果
        for checkpoint in checkpoints:
            # 寫入各類別結果
            for anomaly in anomalies:
                if anomaly in results[checkpoint]:
                    ic_lpips = results[checkpoint][anomaly].get('ic_lpips', 'N/A')
                    is_score = results[checkpoint][anomaly].get('inception_score', 'N/A')

                    # 格式化數值
                    if isinstance(ic_lpips, float):
                        ic_lpips = f"{ic_lpips:.4f}"
                    if isinstance(is_score, float):
                        is_score = f"{is_score:.4f}"

                    writer.writerow([f"checkpoint-{checkpoint}", anomaly, ic_lpips, is_score])

            # 寫入平均值
            avg = averages[checkpoint]
            ic_lpips_avg = f"{avg['ic_lpips_avg']:.4f}" if avg['ic_lpips_avg'] is not None else 'N/A'
            is_avg = f"{avg['is_avg']:.4f}" if avg['is_avg'] is not None else 'N/A'

            writer.writerow([
                f"checkpoint-{checkpoint}",
                f"AVERAGE (n={avg['ic_lpips_count']})",
                ic_lpips_avg,
                is_avg
            ])
            writer.writerow([])  # 空行分隔

        # 寫入總體統計
        writer.writerow(['===== SUMMARY STATISTICS ====='])
        writer.writerow(['Checkpoint', 'Average IC-LPIPS', 'Average Inception Score', 'Sample Count'])
        for checkpoint in checkpoints:
            avg = averages[checkpoint]
            ic_lpips_avg = f"{avg['ic_lpips_avg']:.4f}" if avg['ic_lpips_avg'] is not None else 'N/A'
            is_avg = f"{avg['is_avg']:.4f}" if avg['is_avg'] is not None else 'N/A'
            writer.writerow([f"checkpoint-{checkpoint}", ic_lpips_avg, is_avg, avg['ic_lpips_count']])

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description='彙總評估結果並計算平均值',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:
  # 使用預設值 (hazelnut, dino.5)
  python summarize_results.py

  # 指定生成目錄
  python summarize_results.py --generate-dir /path/to/generate_data/hazelnut

  # 指定異常類型和 checkpoint
  python summarize_results.py --anomalies crack print hole --checkpoints 1000 2000 3000

  # 自訂輸出檔名
  python summarize_results.py --output-name custom_summary.csv
        """
    )

    # 預設值
    default_base_dir = os.path.dirname(os.path.abspath(__file__))
    default_generate_dir = os.path.join(default_base_dir, "generate_data_dino.5", "hazelnut")

    parser.add_argument(
        '--generate-dir',
        type=str,
        default=default_generate_dir,
        help=f'生成數據目錄路徑 (預設: {default_generate_dir})'
    )

    parser.add_argument(
        '--anomalies',
        nargs='+',
        default=["crack", "print", "hole", "cut"],
        help='異常類型列表 (預設: crack print hole cut)'
    )

    parser.add_argument(
        '--checkpoints',
        type=int,
        nargs='+',
        default=[1000, 2000, 3000, 4000, 5000],
        help='Checkpoint 列表 (預設: 1000 2000 3000 4000 5000)'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='輸出檔名 (預設: 自動根據目錄名稱生成)'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # 取得類別名稱（從路徑中提取）
    category_name = os.path.basename(args.generate_dir)

    print("="*80)
    print("評估結果彙總工具")
    print("="*80)
    print(f"生成目錄: {args.generate_dir}")
    print(f"類別名稱: {category_name}")
    print(f"異常類型: {', '.join(args.anomalies)}")
    print(f"Checkpoints: {', '.join(map(str, args.checkpoints))}")
    print("="*80)

    print("\n開始收集評估結果...")
    results = collect_results(args.generate_dir, args.anomalies, args.checkpoints)

    print("計算平均值...")
    averages = calculate_averages(results, args.anomalies, args.checkpoints)

    # 生成彙總報告
    if args.output_name:
        output_csv = os.path.join(args.generate_dir, args.output_name)
    else:
        output_csv = os.path.join(args.generate_dir, f"{category_name}_evaluation_summary.csv")

    print(f"生成彙總報告: {output_csv}")
    generate_summary(results, averages, output_csv, args.anomalies, args.checkpoints)

    # 打印到終端
    print("\n" + "="*80)
    print(f"評估結果彙總 ({category_name})")
    print("="*80)

    for checkpoint in args.checkpoints:
        print(f"\n【Checkpoint-{checkpoint}】")
        for anomaly in args.anomalies:
            if anomaly in results[checkpoint]:
                ic_lpips = results[checkpoint][anomaly].get('ic_lpips', 'N/A')
                is_score = results[checkpoint][anomaly].get('inception_score', 'N/A')

                if isinstance(ic_lpips, float):
                    ic_lpips_str = f"{ic_lpips:.4f}"
                else:
                    ic_lpips_str = str(ic_lpips)

                if isinstance(is_score, float):
                    is_score_str = f"{is_score:.4f}"
                else:
                    is_score_str = str(is_score)

                print(f"  {anomaly:10s} - IC-LPIPS: {ic_lpips_str:>8s}  Inception Score: {is_score_str:>8s}")

        avg = averages[checkpoint]
        if avg['ic_lpips_avg'] is not None:
            print(f"  {'AVERAGE':10s} - IC-LPIPS: {avg['ic_lpips_avg']:8.4f}  Inception Score: {avg['is_avg']:8.4f}  (n={avg['ic_lpips_count']})")

    print("\n" + "="*80)
    print("彙總完成！結果已保存至:", output_csv)
    print("="*80)

if __name__ == "__main__":
    main()
