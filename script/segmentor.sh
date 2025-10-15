export MODEL_NAME="models/u2net.pth"

# 從命令列參數讀取
IMAGE_DIR="$1"
OUTPUT_DIR="$2"

# 顯示使用說明
if [ $# -eq 0 ]; then
    echo "Usage: $0 <input_folder> [output_folder]"
    echo "Example: $0 generate_data/hazelnut/hole"
    echo "Example: $0 datasets/mvtec_ad/hazelnut/test/hole processed_datasets/temp/hazelnut/hole"
    exit 1
fi

# 檢查輸入資料夾是否存在
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Directory '$IMAGE_DIR' does not exist"
    exit 1
fi

# 如果沒有指定輸出目錄，使用預設命名（在同一父目錄下）
if [ -z "$OUTPUT_DIR" ]; then
    FOLDER_NAME=$(basename "$IMAGE_DIR")
    PARENT_DIR=$(dirname "$IMAGE_DIR")
    OUTPUT_DIR="$PARENT_DIR/${FOLDER_NAME}_segmented"
fi

# 建立輸出資料夾
mkdir -p "$OUTPUT_DIR"

echo "Processing images from: $IMAGE_DIR"
echo "Saving results to: $OUTPUT_DIR"

.env/bin/python extra_repository/U-2-Net/u2net_test.py \
    --model_dir=$MODEL_NAME \
    --input_dir="$IMAGE_DIR" \
    --output_dir="$OUTPUT_DIR"