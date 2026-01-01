export MVTEC_NAME="hazelnut"

export MODEL_NAME="models/stable-diffusion-v1-5"

# export ANOMALIES=("crack" "cut")
export ANOMALIES=("hole" "print")

export CHECKPOINT_STEPS=(1000 2000 3000 4000 5000)
# export CHECKPOINT_STEPS=(6000 7000 8000 9000 10000)

for ANOMALIE in "${ANOMALIES[@]}"; do
    export INSTANCE_PROMPT_BLEND="=a sks $NAME with a hta $ANOMALY"
    export INSTANCE_PROMPT_FG="a hta $ANOMALY"

    for CHECKPOINT_STEP in "${CHECKPOINT_STEPS[@]}"; do

        export LORA_WEIGHTS="all_generate_sks_NAME_hta_ANOMALY_mse/$MVTEC_NAME/stage2-$ANOMALIE-dual/checkpoint-$CHECKPOINT_STEP"
        export OUTPUT_DIR="generate_data_sks_NAME_hta_ANOMALY_mse/$MVTEC_NAME/stage2-$ANOMALIE-dual/checkpoint-$CHECKPOINT_STEP"

        echo "=================================================="
        echo "開始 Stage 2 推理：生成異常圖片"
        echo "類別: $MVTEC_NAME"
        echo "異常: $ANOMALIE"
        echo "Prompt Blend: $INSTANCE_PROMPT_BLEND"
        echo "Prompt FG: $INSTANCE_PROMPT_FG"
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