#!/bin/bash

export MODEL_NAME="models/stable-diffusion-v1-5"
export INSTANCE_DIR="datasets/mvtec_ad"
export BASE_DIR="mse_dino"
export INSTANCE_PROMPT_BLEND="a vfx with sks"
export INSTANCE_PROMPT_FG="sks"

export NAME="leather"
export ANOMALIES=("color" "cut" "fold" "glue" "poke")

for ANOMALY in "${ANOMALIES[@]}"; do
    accelerate launch train/stage2-dual/train.py \
        --mixed_precision="no" \
        --train_text_encoder \
        --mvtec_name=$NAME \
        --mvtec_anamaly_name=$ANOMALY \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --instance_data_dir=$INSTANCE_DIR \
        --output_dir="all_generate_$BASE_DIR/$NAME/stage2-$ANOMALY-dual" \
        --instance_prompt_blend="$INSTANCE_PROMPT_BLEND" \
        --instance_prompt_fg="$INSTANCE_PROMPT_FG" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=1 \
        --gradient_checkpointing \
        --learning_rate=2e-5 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=2500 \
        --rank 32 \
        --seed 32 \
        --report_to="tensorboard"

    accelerate launch train/stage2-dual/train.py \
        --mixed_precision="no" \
        --train_text_encoder \
        --mvtec_name=$NAME \
        --mvtec_anamaly_name=$ANOMALY \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --instance_data_dir=$INSTANCE_DIR \
        --output_dir="all_generate_$BASE_DIR/$NAME/stage2-$ANOMALY-dual" \
        --instance_prompt_blend="$INSTANCE_PROMPT_BLEND" \
        --instance_prompt_fg="$INSTANCE_PROMPT_FG" \
        --resolution=512 \
        --train_batch_size=4 \
        --gradient_accumulation_steps=1 \
        --gradient_checkpointing \
        --learning_rate=2e-5 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=5000 \
        --rank 32 \
        --seed 32 \
        --report_to="tensorboard" \
        --use_dinov2_loss \
        --dinov2_loss_weight=1 \
        --dinov2_model_name="vitb14" \
        --dinov2_loss_type="l2" \
        --dinov2_feature_layers 3 6 9 11 \
        --resume_from_checkpoint="latest"

    export LORA_WEIGHTS="all_generate_$BASE_DIR/$NAME/stage2-$ANOMALY-dual/checkpoint-5000"
    export INFERENCE_OUTPUT_DIR="generate_data_$BASE_DIR/$NAME/stage2-$ANOMALY-dual/checkpoint-5000"

    echo "=================================================="
    echo "開始 Stage 2 推理：生成異常圖片"
    echo "類別: $NAME"
    echo "異常: $ANOMALY"
    echo "Prompt Blend: $INSTANCE_PROMPT_BLEND"
    echo "檢查點: checkpoint-5000"
    echo "LORA_WEIGHTS: $LORA_WEIGHTS"
    echo "OUTPUT_DIR: $INFERENCE_OUTPUT_DIR"
    echo "=================================================="

    CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
        --model_name=$MODEL_NAME \
        --lora_weights=$LORA_WEIGHTS \
        --num_images=100 \
        --prompt="$INSTANCE_PROMPT_BLEND" \
        --num_inference_steps=100 \
        --output_dir=$INFERENCE_OUTPUT_DIR \
        --enable_xformers \
        --enable_vae_slicing

    echo "=================================================="
    echo "開始評估：計算 IC-LPIPS 和 Inception Score"
    echo "圖片路徑: $INFERENCE_OUTPUT_DIR/image"
    echo "=================================================="

    bash script/eval/eval_all.sh "$INFERENCE_OUTPUT_DIR/image" "$INSTANCE_DIR" 0

    if [ $? -eq 0 ]; then
        echo "✓ 評估完成"
        echo "  結果位置: $INFERENCE_OUTPUT_DIR/ic_lpips_results.csv"
        echo "  結果位置: $INFERENCE_OUTPUT_DIR/IS_results.csv"
    else
        echo "✗ 評估失敗"
    fi
done
