export MODEL_NAME="/models/u2net.pth"
export INPUT_DIR="/generate_data/hazelnut/hole/fg"
export OUTPUT_DIR="/generate_data/hazelnut/hole/mask"

rm -rf $OUTPUT_DIR

python /U-2-Net/u2net_test.py \
    --model_dir=$MODEL_NAME \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR