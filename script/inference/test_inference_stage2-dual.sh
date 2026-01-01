#!/bin/bash
# 測試 dual inference pipeline - 同時生成瑕疵圖像和前景遮罩

export MVTEC_NAME="hazelnut"
export MODEL_NAME="models/stable-diffusion-v1-5"
export ANOMALIE="hole"
export CHECKPOINT_STEP=5000

# 提示詞策略 B：使用更稀有的 token
export INSTANCE_PROMPT_BLEND="a ohwx with zwx"
export INSTANCE_PROMPT_FG="zwx"

export LORA_WEIGHTS="all_generate_opt_prompt/$MVTEC_NAME/stage2-$ANOMALIE-dual/checkpoint-$CHECKPOINT_STEP"
export OUTPUT_DIR="test"

echo "=================================================="
echo "測試 Dual Inference Pipeline"
echo "類別: $MVTEC_NAME"
echo "異常: $ANOMALIE"
echo "檢查點: checkpoint-$CHECKPOINT_STEP"
echo "LORA_WEIGHTS: $LORA_WEIGHTS"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "=================================================="

CUDA_VISIBLE_DEVICES=0 python inference/inference_dual.py \
    --model_name=$MODEL_NAME \
    --lora_weights=$LORA_WEIGHTS \
    --num_images=3 \
    --prompt_blend="$INSTANCE_PROMPT_BLEND" \
    --prompt_fg="$INSTANCE_PROMPT_FG" \
    --num_inference_steps=100 \
    --guidance_scale=2.5 \
    --output_dir=$OUTPUT_DIR \
    --seed=42 \
    --enable_xformers \
    --enable_vae_slicing
