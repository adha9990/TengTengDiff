export MODEL_NAME="/models/u2net.pth"

export NAME="hazelnut"
export ANOMALY="hole"

export DIR="/generate_data"
export OUTPUT_DIR="$DIR/$NAME/$ANOMALY/mask"

rm -rf $OUTPUT_DIR

python /U-2-Net/u2net_test.py \
    --model_dir=$MODEL_NAME \
    --input_dir="$DIR/$NAME/$ANOMALY/fg" \
    --output_dir="$DIR/$NAME/$ANOMALY/mask"