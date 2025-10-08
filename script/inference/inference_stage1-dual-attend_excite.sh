#!/bin/bash

# Stage 1-Dual + Attend-and-Excite 推論
# 使用 Stage 1-dual 訓練的檢查點，測試 Attend-and-Excite 在早期階段的效果

export MVTEC_NAME="hazelnut"
export MVTEC_ANOMALY_NAME="hole"

export MODEL_NAME="models/stable-diffusion-v1-5"
export LORA_WEIGHTS="all_generate/$MVTEC_NAME/stage1-$MVTEC_ANOMALY_NAME-dual/checkpoint-5000"
export OUTPUT_DIR="generate_data/$MVTEC_NAME/stage1-$MVTEC_ANOMALY_NAME-dual-attend_excite"

export INSTANCE_PROMPT_BLEND="a vfx with sks"
export INSTANCE_PROMPT_FG="sks"

# Attend-and-Excite 參數說明：
# 對於 Stage 1-dual ("a vfx with sks" 提示詞)：
#   - token 0: [CLS]
#   - token 1: a
#   - token 2: vfx (正常物體外觀)
#   - token 3: with
#   - token 4: sks (在 stage1-dual 中，這個 token 與正常圖片一起訓練)
#   - token 5: [SEP]
#
# 實驗目的：
#   測試 Attend-and-Excite 在 Stage 1-dual 階段的效果
#   理論上 "sks" token 在這個階段還沒有真正學到異常特徵
#   但強化它可能會影響生成的視覺特性

echo "========================================="
echo "Stage 1-Dual + Attend-and-Excite 推論"
echo "========================================="
echo "檢查點: $LORA_WEIGHTS"
echo "輸出目錄: $OUTPUT_DIR"
echo "提示詞: $INSTANCE_PROMPT_BLEND"
echo "強化 token: 4 (sks)"
echo "========================================="

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
echo "Stage 1-Dual + Attend-and-Excite 推論完成"
echo "輸出目錄: $OUTPUT_DIR"
echo "========================================="
