#!/bin/bash

export MODEL_NAME="models/stable-diffusion-v1-5"
export INSTANCE_DIR="datasets/mvtec_ad"

export NAME="hazelnut"

export OUTPUT_DIR="all_generate_sks_NAME_hta_ANOMALY_mse/"

ANOMALIES=("crack" "cut" "print" "hole")

for ANOMALY in "${ANOMALIES[@]}"; do

    export INSTANCE_PROMPT_BLEND="=a sks $NAME with a hta $ANOMALY"
    export INSTANCE_PROMPT_FG="a hta $ANOMALY"

    accelerate launch train/stage2-dual/train.py \
        --mixed_precision="no" \
        --mvtec_name=$NAME \
        --mvtec_anamaly_name=$ANOMALY \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --instance_data_dir=$INSTANCE_DIR \
        --output_dir="$OUTPUT_DIR/$NAME/stage2-$ANOMALY-dual" \
        --instance_prompt_blend="$INSTANCE_PROMPT_BLEND" \
        --instance_prompt_fg="$INSTANCE_PROMPT_FG" \
        --resolution=512 \
        --train_batch_size=8 \
        --gradient_accumulation_steps=32 \
        --gradient_checkpointing \
        --learning_rate=2e-5 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=5000 \
        --rank 32 \
        --seed 32 \
        --report_to="tensorboard"
    
    # accelerate launch train/stage2-dual/train.py \
    #     --mixed_precision="no" \
    #     --mvtec_name=$NAME \
    #     --mvtec_anamaly_name=$ANOMALY \
    #     --pretrained_model_name_or_path=$MODEL_NAME \
    #     --instance_data_dir=$INSTANCE_DIR \
    #     --output_dir="$OUTPUT_DIR/$NAME/stage2-$ANOMALY-dual" \
    #     --instance_prompt_blend="$INSTANCE_PROMPT_BLEND" \
    #     --instance_prompt_fg="$INSTANCE_PROMPT_FG" \
    #     --resolution=512 \
    #     --train_batch_size=1 \
    #     --gradient_accumulation_steps=64 \
    #     --gradient_checkpointing \
    #     --learning_rate=2e-5 \
    #     --lr_scheduler="constant" \
    #     --lr_warmup_steps=0 \
    #     --max_train_steps=5000 \
    #     --rank 32 \
    #     --seed 32 \
    #     --report_to="tensorboard" \
    #     --use_dinov2_loss \
    #     --dinov2_loss_weight=1 \
    #     --dinov2_model_name="vitb14" \
    #     --dinov2_loss_type="l2" \
    #     --dinov2_feature_layers 3 6 9 11 \
    #     --resume_from_checkpoint="latest"
done