#!/bin/bash

export MODEL_NAME="models/stable-diffusion-v1-5"
export INSTANCE_DIR="datasets/mvtec_ad"

export NAME="hazelnut"

export INSTANCE_PROMPT="a vfx"

export OUTPUT_DIR="all_generate/"

accelerate launch train/stage1/train.py \
    --mixed_precision="no" \
    --mvtec_name=$NAME \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir="$OUTPUT_DIR/$NAME/full" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --resolution=512 \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=5000 \
    --rank 32 \
    --seed 32 \
    --num_validation_images=4 \
    --validation_prompt="$INSTANCE_PROMPT" \
    --train_text_encoder \
    --num_inference_steps=25 \
    --enable_xformers_memory_efficient_attention \
    --report_to="tensorboard"