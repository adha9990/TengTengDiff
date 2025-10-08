#!/bin/bash

# Stage 2 推論使用 Attend-and-Excite 增強異常生成

export MVTEC_NAME="hazelnut"
export MVTEC_ANOMALY_NAME="hole"

export MODEL_NAME="models/stable-diffusion-v1-5"
export LORA_WEIGHTS="all_generate/$MVTEC_NAME/stage2-$MVTEC_ANOMALY_NAME/checkpoint-8000"
export OUTPUT_DIR="generate_data/$MVTEC_NAME/stage2-$MVTEC_ANOMALY_NAME-attend_excite"

export INSTANCE_PROMPT_BLEND="a vfx with sks"
export INSTANCE_PROMPT_FG="sks"

# Attend-and-Excite 參數說明：
# --token_indices: 要強化的 token 索引（通常是異常相關的 token）
#   對於 "a vfx with sks" 提示詞：
#   - token 0: [CLS]
#   - token 1: a
#   - token 2: vfx
#   - token 3: with
#   - token 4: sks
#   - token 5: [SEP]
#   因此 "sks" 的索引是 4
#
# --max_iter_to_alter: 應用 Attend-and-Excite 的最大去噪步數（預設 25）
# --scale_factor: latent 更新的縮放因子（預設 20，可調整至 30-40 以增強效果）
# --attention_res: 注意力圖解析度（預設 16）
# --smooth_attentions: 啟用高斯平滑以減少噪聲
# --sigma: 高斯平滑的標準差（預設 0.5）

python inference/inference-attend_and_excite.py \
    --model_name=$MODEL_NAME \
    --lora_weights=$LORA_WEIGHTS \
    --num_images=100 \
    --prompt="$INSTANCE_PROMPT_BLEND" \
    --num_inference_steps=50 \
    --guidance_scale=7.5 \
    --output_dir=$OUTPUT_DIR \
    --token_indices="4" \
    --max_iter_to_alter=25 \
    --scale_factor=20 \
    --attention_res=16 \
    --smooth_attentions \
    --sigma=0.5 \
    --kernel_size=3 \
    --enable_xformers \
    --enable_vae_slicing
