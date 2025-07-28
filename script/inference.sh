export MVTEC_NAME="hazelnut"
export MVTEC_ANOMALY_NAME="hole"

export MODEL_NAME="models/stable-diffusion-v1-5"
export LORA_WEIGHTS="all_generate/$MVTEC_NAME/$MVTEC_ANOMALY_NAME/checkpoint-1000"
export OUTPUT_DIR="generate_data/$MVTEC_NAME/$MVTEC_ANOMALY_NAME"

rm -rf generate_data/

python inference.py \
    --model_name=$MODEL_NAME \
    --lora_weights=$LORA_WEIGHTS \
    --mvtec_name=$MVTEC_NAME \
    --mvtec_aomaly_name=$MVTEC_ANOMALY_NAME \
    --num_images=100 \
    --prompt_blend="a vfx with sks" \
    --prompt_fg="sks" \
    --num_inference_steps=100 \
    --output_dir=$OUTPUT_DIR