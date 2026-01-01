export MODEL_NAME="models/stable-diffusion-v1-5"

export MVTEC_NAME="hazelnut"

# 提示詞策略 B：使用更稀有的 token
export INSTANCE_PROMPT="a ohwx"

export CHECKPOINT_STEPS=(5000)

for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do

    echo "=================================================="
    echo "開始 Stage 1 推理：生成正常圖片"
    echo "類別: $MVTEC_NAME"
    echo "檢查點: checkpoint-$CHECKPOINT_STEP"
    echo "=================================================="

    export LORA_WEIGHTS="all_generate/$MVTEC_NAME/stage1-full/checkpoint-$CHECKPOINT_STEP"
    export OUTPUT_DIR="generate_data/$MVTEC_NAME/stage1-full/checkpoint-$CHECKPOINT_STEP"

    CUDA_VISIBLE_DEVICES=0 python inference/inference.py \
        --model_name=$MODEL_NAME \
        --lora_weights=$LORA_WEIGHTS \
        --num_images=100 \
        --prompt="$INSTANCE_PROMPT" \
        --num_inference_steps=50 \
        --output_dir=$OUTPUT_DIR \
        --enable_xformers \
        --enable_vae_slicing
done