#!/bin/bash

export MODEL_NAME="models/stable-diffusion-v1-5"
export INSTANCE_DIR="datasets/mvtec_ad"

export NAME="hazelnut"

# 提示詞策略 B：使用更稀有的 token 避免語義衝突
# - ohwx: 完全無語義的稀有 token（替代 vfx）
# - 原因：vfx 可能與 "visual effects" 有弱關聯
export INSTANCE_PROMPT="a ohwx"

export OUTPUT_DIR="all_generate/"

accelerate launch train/stage1-full/train.py \
    --mixed_precision="no" \
    --mvtec_name=$NAME \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir="$OUTPUT_DIR/$NAME/stage1-full" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --resolution=512 \
    --train_batch_size=2 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=5000 \
    --rank 32 \
    --seed 32 \
    --num_inference_steps=25 \
    --report_to="tensorboard"
    # --use_dinov2_loss \
    # --dinov2_loss_weight=0.1 \
    # --dinov2_model_name="vitb14" \
    # --dinov2_loss_type="l2" \
    # --dinov2_feature_layers 3 6 9 11