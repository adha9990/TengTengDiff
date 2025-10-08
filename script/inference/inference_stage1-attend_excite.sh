#!/bin/bash

# Stage 1 推論使用 Attend-and-Excite 增強正常圖片生成

export MVTEC_NAME="hazelnut"

export MODEL_NAME="models/stable-diffusion-v1-5"
export LORA_WEIGHTS="all_generate/$MVTEC_NAME/stage1-full/checkpoint-5000"
export OUTPUT_DIR="generate_data/$MVTEC_NAME/stage1-full-attend_excite"

export INSTANCE_PROMPT="a vfx"

# Attend-and-Excite 參數說明：
# 對於 Stage 1 ("a vfx" 提示詞)：
#   - token 0: [CLS]
#   - token 1: a
#   - token 2: vfx
#   - token 3: [SEP]
#   因此 "vfx" 的索引是 2
#
# 這裡我們強化 "vfx" token 以確保物體特徵被正確生成

python inference/inference-attend_and_excite.py \
    --model_name=$MODEL_NAME \
    --lora_weights=$LORA_WEIGHTS \
    --num_images=100 \
    --prompt="$INSTANCE_PROMPT" \
    --num_inference_steps=50 \
    --guidance_scale=7.5 \
    --output_dir=$OUTPUT_DIR \
    --token_indices="2" \
    --max_iter_to_alter=25 \
    --scale_factor=20 \
    --attention_res=16 \
    --smooth_attentions \
    --sigma=0.5 \
    --kernel_size=3 \
    --enable_xformers \
    --enable_vae_slicing
