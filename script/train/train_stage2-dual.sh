#!/bin/bash

export MODEL_NAME="models/stable-diffusion-v1-5"
export INSTANCE_DIR="datasets/mvtec_ad"

export NAME="hazelnut"

export INSTANCE_PROMPT_BLEND="a vfx with sks"
export INSTANCE_PROMPT_FG="sks"

export OUTPUT_DIR="all_generate/"

# ANOMALIES=("crack" "cut" "print" "hole")
ANOMALIES=("hole")

for ANOMALY in "${ANOMALIES[@]}"; do
    accelerate launch train/stage2-dual/train.py \
        --mixed_precision="no" \
        --mvtec_name=$NAME \
        --mvtec_anamaly_name=$ANOMALY \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --instance_data_dir=$INSTANCE_DIR \
        --output_dir="$OUTPUT_DIR/$NAME/stage2-$ANOMALY-dual_test" \
        --instance_prompt_blend="$INSTANCE_PROMPT_BLEND" \
        --instance_prompt_fg="$INSTANCE_PROMPT_FG" \
        --resolution=512 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=64 \
        --gradient_checkpointing \
        --learning_rate=2e-5 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=10000 \
        --rank 32 \
        --seed 32 \
        --report_to="tensorboard" \
        --resume_from_checkpoint="$OUTPUT_DIR/$NAME/stage2-$ANOMALY-dual_test/checkpoint-6000"
    
    # accelerate launch train/stage2-dual/train.py \
    #     --mixed_precision="no" \
    #     --mvtec_name=$NAME \
    #     --mvtec_anamaly_name=$ANOMALY \
    #     --pretrained_model_name_or_path=$MODEL_NAME \
    #     --instance_data_dir=$INSTANCE_DIR \
    #     --output_dir="$OUTPUT_DIR/$NAME/stage1-$ANOMALY-dual" \
    #     --instance_prompt_blend="$INSTANCE_PROMPT_BLEND" \
    #     --instance_prompt_fg="$INSTANCE_PROMPT_FG" \
    #     --resolution=512 \
    #     --train_batch_size=8 \
    #     --gradient_accumulation_steps=32 \
    #     --gradient_checkpointing \
    #     --learning_rate=2e-5 \
    #     --lr_scheduler="constant" \
    #     --lr_warmup_steps=0 \
    #     --max_train_steps=8000 \
    #     --rank 32 \
    #     --seed 32 \
    #     --report_to="tensorboard" \
    #     --resume_from_checkpoint="latest"
done