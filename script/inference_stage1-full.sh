export MODEL_NAME="models/stable-diffusion-v1-5"

export MVTEC_NAME="hazelnut"

export LORA_WEIGHTS="all_generate/$MVTEC_NAME/stage1-full/checkpoint-5000"
export OUTPUT_DIR="generate_data/$MVTEC_NAME/stage1-full"

export INSTANCE_PROMPT="a vfx"

python inference/inference.py \
    --model_name=$MODEL_NAME \
    --lora_weights=$LORA_WEIGHTS \
    --num_images=100 \
    --prompt="$INSTANCE_PROMPT" \
    --num_inference_steps=50 \
    --output_dir=$OUTPUT_DIR \
    --enable_xformers \
    --enable_vae_slicing