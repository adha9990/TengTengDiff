export MODEL_NAME="models/stable-diffusion-v1-5"

export MVTEC_NAME="hazelnut"

export INSTANCE_PROMPT="a vfx"

export CHECKPOINT_STEPS=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)

for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do

    echo "=================================================="
    echo "開始 Stage 1 推理：生成正常圖片"
    echo "類別: $MVTEC_NAME"
    echo "檢查點: checkpoint-$CHECKPOINT_STEP"
    echo "=================================================="

    export LORA_WEIGHTS="all_generate/$MVTEC_NAME/stage1-full-16/checkpoint-$CHECKPOINT_STEP"
    export OUTPUT_DIR="generate_data/$MVTEC_NAME/stage1-full-16/checkpoint-$CHECKPOINT_STEP"

    python inference/inference.py \
        --model_name=$MODEL_NAME \
        --lora_weights=$LORA_WEIGHTS \
        --num_images=10 \
        --prompt="$INSTANCE_PROMPT" \
        --num_inference_steps=50 \
        --output_dir=$OUTPUT_DIR \
        --enable_xformers \
        --enable_vae_slicing
done