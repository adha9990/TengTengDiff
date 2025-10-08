#!/bin/bash

# Stage 2 Dual 推論使用 Attend-and-Excite
# 這個腳本使用 Stage 2-dual 訓練的檢查點，並應用 Attend-and-Excite 來增強異常生成

export MVTEC_NAME="hazelnut"
export MVTEC_ANOMALY_NAME="hole"

# 使用 GPU 1 避免記憶體衝突
# export CUDA_VISIBLE_DEVICES=1

export MODEL_NAME="models/stable-diffusion-v1-5"
export LORA_WEIGHTS="all_generate/$MVTEC_NAME/stage2-$MVTEC_ANOMALY_NAME-dual/checkpoint-8000"
export OUTPUT_DIR="generate_data/$MVTEC_NAME/stage2-$MVTEC_ANOMALY_NAME-dual-attend_excite"

export INSTANCE_PROMPT_BLEND="a vfx with sks"
export INSTANCE_PROMPT_FG="sks"

# Attend-and-Excite 參數說明：
# 對於 "a vfx with sks" 提示詞：
#   - token 0: [CLS]
#   - token 1: a
#   - token 2: vfx (正常物體外觀)
#   - token 3: with
#   - token 4: sks (異常特徵)
#   - token 5: [SEP]
#
# 策略選項：
# 1. 僅強化異常 token: "4"
# 2. 同時強化物體和異常: "2,4"
# 3. 更強的異常強化（增加 scale_factor）

# 使用策略 1：僅強化異常特徵
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

echo "========================================="
echo "Stage 2-dual + Attend-and-Excite 推論完成"
echo "輸出目錄: $OUTPUT_DIR"
echo "========================================="
