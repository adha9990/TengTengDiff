export MODEL_NAME="models/u2net.pth"

# 從命令列參數讀取
IMAGE_DIR="$1"

# 顯示使用說明
if [ $# -eq 0 ]; then
    echo "Usage: $0 <name> <anomaly>"
    echo "Example: $0 hazelnut hole"
    echo "Using defaults: IMAGE_DIR=$IMAGE_DIR"
fi

python extra_repository/U-2-Net/u2net_test.py \
    --model_dir=$MODEL_NAME \
    --input_dir="$IMAGE_DIR" \
    --output_dir="$IMAGE_DIR/mask"