#!/bin/bash

# ============================================================================
# 結合 Stage 1 和 Stage 2 訓練與推理腳本
# 功能：
# 1. 執行 Stage 1 訓練（正常圖片）
# 2. 執行 Stage 1 推理（生成正常圖片）
# 3. 針對所有異常類型執行 Stage 2 訓練與推理
# ============================================================================

set -e  # 遇到錯誤立即退出

export MODEL_NAME="models/stable-diffusion-v1-5"
export INSTANCE_DIR="datasets/mvtec_ad"

export NAME="hazelnut"
export OUTPUT_DIR="all_generate/"

# 定義所有異常類型
ANOMALIES=("crack" "cut" "print" "hole")

# ============================================================================
# Stage 1: 訓練正常圖片
# ============================================================================
echo "=================================================="
echo "開始 Stage 1 訓練：正常圖片"
echo "類別: $NAME"
echo "=================================================="

export INSTANCE_PROMPT="a vfx"

accelerate launch train/stage1-full/train.py \
    --mixed_precision="no" \
    --mvtec_name=$NAME \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir="$OUTPUT_DIR/$NAME/stage1-full" \
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
    --train_text_encoder \
    --num_inference_steps=25 \
    --report_to="tensorboard"

echo ""
echo "=================================================="
echo "✅ Stage 1 訓練完成"
echo "=================================================="
echo ""

# ============================================================================
# Stage 1 推理：生成正常圖片
# ============================================================================
echo "=================================================="
echo "開始 Stage 1 推理：生成正常圖片"
echo "類別: $NAME"
echo "=================================================="

export STAGE1_LORA_WEIGHTS="$OUTPUT_DIR/$NAME/stage1-full/checkpoint-5000"
export STAGE1_OUTPUT_DIR="generate_data/$NAME/stage1-full"

# 檢查 Stage 1 LoRA 權重是否存在
if [ ! -d "$STAGE1_LORA_WEIGHTS" ]; then
    echo "❌ 錯誤: Stage 1 LoRA 權重不存在: $STAGE1_LORA_WEIGHTS"
    exit 1
fi

.env/bin/python inference/inference.py \
    --model_name=$MODEL_NAME \
    --lora_weights=$STAGE1_LORA_WEIGHTS \
    --num_images=100 \
    --prompt="$INSTANCE_PROMPT" \
    --num_inference_steps=50 \
    --output_dir=$STAGE1_OUTPUT_DIR \
    --enable_xformers \
    --enable_vae_slicing

echo ""
echo "=================================================="
echo "✅ Stage 1 推理完成"
echo "生成圖片: $STAGE1_OUTPUT_DIR"
echo "=================================================="
echo ""

# ============================================================================
# Stage 2: 針對所有異常類型進行訓練與推理
# ============================================================================

export INSTANCE_PROMPT_BLEND="a vfx with sks"
export INSTANCE_PROMPT_FG="sks"
export STAGE1_CHECKPOINT="$OUTPUT_DIR/$NAME/stage1-full/checkpoint-5000"

# 檢查 Stage 1 檢查點是否存在
if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "❌ 錯誤: Stage 1 檢查點不存在: $STAGE1_CHECKPOINT"
    exit 1
fi

for ANOMALY in "${ANOMALIES[@]}"; do
    echo "=================================================="
    echo "開始 Stage 2 訓練"
    echo "類別: $NAME"
    echo "異常類型: $ANOMALY"
    echo "從檢查點恢復: $STAGE1_CHECKPOINT"
    echo "=================================================="

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
        --train_batch_size=4 \
        --gradient_accumulation_steps=1 \
        --learning_rate=2e-5 \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --max_train_steps=8000 \
        --resume_from_checkpoint="$STAGE1_CHECKPOINT" \
        --rank 32 \
        --seed 32 \
        --train_text_encoder \
        --num_inference_steps=25 \
        --report_to="tensorboard"

    echo ""
    echo "=================================================="
    echo "✅ Stage 2 訓練完成：$ANOMALY"
    echo "=================================================="
    echo ""

    # ------------------------------------------------------------------------
    # Stage 2 推理：生成異常圖片
    # ------------------------------------------------------------------------
    echo "=================================================="
    echo "開始 Stage 2 推理：$ANOMALY"
    echo "類別: $NAME"
    echo "異常類型: $ANOMALY"
    echo "=================================================="

    export STAGE2_LORA_WEIGHTS="$OUTPUT_DIR/$NAME/stage2-$ANOMALY-dual/checkpoint-8000"
    export STAGE2_OUTPUT_DIR="generate_data/$NAME/stage2-$ANOMALY-dual"

    # 檢查 Stage 2 LoRA 權重是否存在
    if [ ! -d "$STAGE2_LORA_WEIGHTS" ]; then
        echo "❌ 錯誤: Stage 2 LoRA 權重不存在: $STAGE2_LORA_WEIGHTS"
        exit 1
    fi

    .env/bin/python inference/inference.py \
        --model_name=$MODEL_NAME \
        --lora_weights=$STAGE2_LORA_WEIGHTS \
        --num_images=100 \
        --prompt="$INSTANCE_PROMPT_BLEND" \
        --num_inference_steps=50 \
        --output_dir=$STAGE2_OUTPUT_DIR \
        --enable_xformers \
        --enable_vae_slicing

    echo ""
    echo "=================================================="
    echo "✅ Stage 2 推理完成：$ANOMALY"
    echo "生成圖片: $STAGE2_OUTPUT_DIR"
    echo "=================================================="
    echo ""
done

# ============================================================================
# 完成總結
# ============================================================================
echo "=================================================="
echo "🎉 所有訓練與推理完成！"
echo "=================================================="
echo "類別: $NAME"
echo ""
echo "📦 Stage 1 檢查點:"
echo "  $OUTPUT_DIR/$NAME/stage1-full/checkpoint-5000"
echo ""
echo "🖼️ Stage 1 生成圖片:"
echo "  generate_data/$NAME/stage1-full/ (100 張正常圖片)"
echo ""
echo "📦 Stage 2 檢查點:"
for ANOMALY in "${ANOMALIES[@]}"; do
    echo "  - $ANOMALY: $OUTPUT_DIR/$NAME/stage2-$ANOMALY-dual/checkpoint-8000"
done
echo ""
echo "🖼️ Stage 2 生成圖片:"
for ANOMALY in "${ANOMALIES[@]}"; do
    echo "  - $ANOMALY: generate_data/$NAME/stage2-$ANOMALY-dual/ (100 張異常圖片)"
done
echo "=================================================="
