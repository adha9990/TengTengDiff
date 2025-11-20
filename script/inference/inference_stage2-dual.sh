export MVTEC_NAME="hazelnut"

export MODEL_NAME="models/stable-diffusion-v1-5"

export INSTANCE_PROMPT_BLEND="a vfx with sks"
export INSTANCE_PROMPT_FG="sks"

# export ANOMALIES=("crack" "cut")
export ANOMALIES=("hole" "print")

export CHECKPOINT_STEPS=(1000 2000 3000 4000 5000)

for ANOMALIE in "${ANOMALIES[@]}"; do
    for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do
    
        export LORA_WEIGHTS="all_generate.2.5k_dino-5k_sme/$MVTEC_NAME/stage1-$ANOMALIE-dual/checkpoint-$CHECKPOINT_STEP"
        export OUTPUT_DIR="generate_data.2.5k_dino-5k_sme/$MVTEC_NAME/stage1-$ANOMALIE-dual/checkpoint-$CHECKPOINT_STEP"

        echo "=================================================="
        echo "開始 Stage 2 推理：生成異常圖片"
        echo "類別: $MVTEC_NAME"
        echo "異常: $ANOMALIE"
        echo "檢查點: checkpoint-$CHECKPOINT_STEP"
        echo "LORA_WEIGHTS: $LORA_WEIGHTS"
        echo "OUTPUT_DIR: $OUTPUT_DIR"
        echo "=================================================="

        CUDA_VISIBLE_DEVICES=1 python inference/inference.py \
            --model_name=$MODEL_NAME \
            --lora_weights=$LORA_WEIGHTS \
            --num_images=100 \
            --prompt="$INSTANCE_PROMPT_BLEND" \
            --num_inference_steps=100 \
            --output_dir=$OUTPUT_DIR \
            --enable_xformers \
            --enable_vae_slicing
    done
done