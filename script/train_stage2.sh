#!/bin/bash

export MODEL_NAME="models/stable-diffusion-v1-5"
export INSTANCE_DIR="datasets/mvtec_ad"

export NAME="hazelnut"
export ANOMALY="hole"

export INSTANCE_PROMPT_BLEND="a vfx with sks"
export INSTANCE_PROMPT_FG="sks"

export OUTPUT_DIR="all_generate/"

accelerate launch src/stage2/train.py \
    --mixed_precision="no" \
    --mvtec_name=$NAME \
    --mvtec_anamaly_name=$ANOMALY \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir="$OUTPUT_DIR/$NAME/$ANOMALY" \
    --instance_prompt_blend="$INSTANCE_PROMPT_BLEND" \
    --instance_prompt_fg="$INSTANCE_PROMPT_FG" \
    --resolution=512 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=8000 \
    --resume_from_checkpoint="$OUTPUT_DIR/$NAME/full/checkpoint-5000" \
    --rank 32 \
    --seed 32 \
    --train_text_encoder \
    --num_inference_steps=25 \
    --report_to="tensorboard"