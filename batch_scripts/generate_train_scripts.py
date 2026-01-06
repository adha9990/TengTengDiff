#!/usr/bin/env python3
from pathlib import Path

# 固定變數
MODEL_NAME = "models/stable-diffusion-v1-5"
INSTANCE_DIR = "datasets/mvtec_ad"
BASE_DIR = "mse_dino"
INSTANCE_PROMPT_BLEND = "a vfx with sks"
INSTANCE_PROMPT_FG = "sks"

# 讀取資料集結構
dataset_path = Path("/home/nknul40s/bluestar-research/TengTengDiff/datasets/mvtec_ad")
category_configs = {}

for category_dir in sorted(dataset_path.iterdir()):
    if not category_dir.is_dir():
        continue

    test_dir = category_dir / "test"
    if test_dir.exists():
        defects = [d.name for d in sorted(test_dir.iterdir())
                   if d.is_dir() and d.name != "good"]
        if defects:
            category_configs[category_dir.name] = defects

# 生成訓練腳本模板
def generate_script(category, anomalies):
    script = f"""#!/bin/bash

export MODEL_NAME="{MODEL_NAME}"
export INSTANCE_DIR="{INSTANCE_DIR}"
export BASE_DIR="{BASE_DIR}"
export INSTANCE_PROMPT_BLEND="{INSTANCE_PROMPT_BLEND}"
export INSTANCE_PROMPT_FG="{INSTANCE_PROMPT_FG}"

export NAME="{category}"
export ANOMALIES=({' '.join([f'"{a}"' for a in anomalies])})

for ANOMALY in "${{ANOMALIES[@]}}"; do
    accelerate launch train/stage2-dual/train.py \\
        --mixed_precision="no" \\
        --train_text_encoder \\
        --mvtec_name=$NAME \\
        --mvtec_anamaly_name=$ANOMALY \\
        --pretrained_model_name_or_path=$MODEL_NAME \\
        --instance_data_dir=$INSTANCE_DIR \\
        --output_dir="all_generate_$BASE_DIR/$NAME/stage2-$ANOMALY-dual" \\
        --instance_prompt_blend="$INSTANCE_PROMPT_BLEND" \\
        --instance_prompt_fg="$INSTANCE_PROMPT_FG" \\
        --resolution=512 \\
        --train_batch_size=4 \\
        --gradient_accumulation_steps=1 \\
        --gradient_checkpointing \\
        --learning_rate=2e-5 \\
        --lr_scheduler="constant" \\
        --lr_warmup_steps=0 \\
        --max_train_steps=2500 \\
        --rank 32 \\
        --seed 32 \\
        --report_to="tensorboard"

    accelerate launch train/stage2-dual/train.py \\
        --mixed_precision="no" \\
        --train_text_encoder \\
        --mvtec_name=$NAME \\
        --mvtec_anamaly_name=$ANOMALY \\
        --pretrained_model_name_or_path=$MODEL_NAME \\
        --instance_data_dir=$INSTANCE_DIR \\
        --output_dir="all_generate_$BASE_DIR/$NAME/stage2-$ANOMALY-dual" \\
        --instance_prompt_blend="$INSTANCE_PROMPT_BLEND" \\
        --instance_prompt_fg="$INSTANCE_PROMPT_FG" \\
        --resolution=512 \\
        --train_batch_size=4 \\
        --gradient_accumulation_steps=1 \\
        --gradient_checkpointing \\
        --learning_rate=2e-5 \\
        --lr_scheduler="constant" \\
        --lr_warmup_steps=0 \\
        --max_train_steps=5000 \\
        --rank 32 \\
        --seed 32 \\
        --report_to="tensorboard" \\
        --use_dinov2_loss \\
        --dinov2_loss_weight=1 \\
        --dinov2_model_name="vitb14" \\
        --dinov2_loss_type="l2" \\
        --dinov2_feature_layers 3 6 9 11 \\
        --resume_from_checkpoint="latest"

    export LORA_WEIGHTS="all_generate_$BASE_DIR/$NAME/stage2-$ANOMALY-dual/checkpoint-5000"
    export INFERENCE_OUTPUT_DIR="generate_data_$BASE_DIR/$NAME/stage2-$ANOMALY-dual/checkpoint-5000"

    echo "=================================================="
    echo "開始 Stage 2 推理：生成異常圖片"
    echo "類別: $NAME"
    echo "異常: $ANOMALY"
    echo "Prompt Blend: $INSTANCE_PROMPT_BLEND"
    echo "檢查點: checkpoint-5000"
    echo "LORA_WEIGHTS: $LORA_WEIGHTS"
    echo "OUTPUT_DIR: $INFERENCE_OUTPUT_DIR"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=0 python inference/inference.py \\
        --model_name=$MODEL_NAME \\
        --lora_weights=$LORA_WEIGHTS \\
        --num_images=100 \\
        --prompt="$INSTANCE_PROMPT_BLEND" \\
        --num_inference_steps=100 \\
        --output_dir=$INFERENCE_OUTPUT_DIR \\
        --enable_xformers \\
        --enable_vae_slicing

    echo "=================================================="
    echo "開始評估：計算 IC-LPIPS 和 Inception Score"
    echo "圖片路徑: $INFERENCE_OUTPUT_DIR/image"
    echo "=================================================="

    bash script/eval/eval_all.sh "$INFERENCE_OUTPUT_DIR/image" "$INSTANCE_DIR" 0

    if [ $? -eq 0 ]; then
        echo "✓ 評估完成"
        echo "  結果位置: $INFERENCE_OUTPUT_DIR/ic_lpips_results.csv"
        echo "  結果位置: $INFERENCE_OUTPUT_DIR/IS_results.csv"
    else
        echo "✗ 評估失敗"
    fi
done
"""
    return script

# 創建輸出目錄
output_dir = Path("batch_scripts")
output_dir.mkdir(exist_ok=True)

print(f"找到 {len(category_configs)} 個類別\n")

# 生成各類別訓練腳本
script_files = []
for category, anomalies in category_configs.items():
    script_path = output_dir / f"train_{category}.sh"
    script_files.append(script_path)

    with open(script_path, 'w') as f:
        f.write(generate_script(category, anomalies))

    script_path.chmod(0o755)
    print(f"✓ {script_path.name} ({len(anomalies)} 個瑕疵)")

# 生成批次執行腳本
batch_script = """#!/bin/bash

# 批次執行所有訓練腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"""

for script_file in script_files:
    batch_script += f"""
echo "========================================"
echo "開始訓練: {script_file.stem.replace('train_', '')}"
echo "========================================"
bash "$SCRIPT_DIR/{script_file.name}"

"""

batch_script += """
echo "========================================"
echo "所有訓練完成"
echo "========================================"
"""

batch_script_path = output_dir / "batch_train_all.sh"
with open(batch_script_path, 'w') as f:
    f.write(batch_script)

batch_script_path.chmod(0o755)

print(f"\n✓ {batch_script_path.name} (批次執行腳本)")
print(f"\n生成完成！所有腳本位於: {output_dir}")
print(f"\n執行方式:")
print(f"  單一類別: bash {output_dir}/train_<類別名>.sh")
print(f"  全部類別: bash {output_dir}/batch_train_all.sh")
