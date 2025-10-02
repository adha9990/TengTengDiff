export MVTEC_NAME="hazelnut"
export MVTEC_ANOMALY_NAME="hole"

export MODEL_NAME="models/stable-diffusion-v1-5"
export LORA_WEIGHTS="all_generate/$MVTEC_NAME/stage1-$MVTEC_ANOMALY_NAME-dual/checkpoint-5000"
export OUTPUT_DIR="generate_data/$MVTEC_NAME/stage1-$MVTEC_ANOMALY_NAME-dual"

export INSTANCE_PROMPT_BLEND="a vfx with sks"
export INSTANCE_PROMPT_FG="sks"

python inference/inference.py \
    --model_name=$MODEL_NAME \
    --lora_weights=$LORA_WEIGHTS \
    --num_images=100 \
    --prompt="$INSTANCE_PROMPT_BLEND" \
    --num_inference_steps=50 \
    --output_dir=$OUTPUT_DIR \
    --enable_xformers \
    --enable_vae_slicing