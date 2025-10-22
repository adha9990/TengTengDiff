#!/bin/bash

# Cross-Attention Map 視覺化腳本
# 用於視覺化訓練後的 LoRA 模型的 attention maps

# ==================== 配置參數 ====================

# MVTec-AD 類別名稱（修改這裡）
NAME="hazelnut"

# 異常類型（可選，如果是 Stage 1 則留空）
ANOMALY="hole"  # 例如: "crack", "hole", "scratch" 或留空為 Stage 1

# 檢查點步
CHECKPOINT_STEP="5000"  # Stage 1 通常是 5000，Stage 2 通常是 8000

# 提示詞
PROMPT="a vfx"  # Stage 1: "a vfx", Stage 2: "a vfx with sks"

# Attention 解析度（16 或 32，越大越精細但越慢）
ATTENTION_RES=16

# 推論步數
NUM_INFERENCE_STEPS=50

# 隨機種子
SEED=42

# Guidance scale
GUIDANCE_SCALE=7.5

# ==================== 自動路徑配置 ====================

# 基礎模型路徑
MODEL_NAME="models/stable-diffusion-v1-5"

LORA_WEIGHTS="all_generate/${NAME}/stage1-full/checkpoint-${CHECKPOINT_STEP}"
OUTPUT_DIR="attention_maps/${NAME}/stage1-full/checkpoint-${CHECKPOINT_STEP}"

# ==================== 檢查和執行 ====================

echo "================================================"
echo "Cross-Attention Map 視覺化"
echo "================================================"
echo "類別: $NAME"
if [ -n "$ANOMALY" ]; then
    echo "異常類型: $ANOMALY"
else
    echo "階段: Stage 1 (正常訓練)"
fi
echo "檢查點: checkpoint-${CHECKPOINT_STEP}"
echo "提示詞: $PROMPT"
echo "輸出目錄: $OUTPUT_DIR"
echo "================================================"

# 檢查 LoRA 權重是否存在
if [ ! -d "$LORA_WEIGHTS" ]; then
    echo "❌ 錯誤: LoRA 權重不存在: $LORA_WEIGHTS"
    echo "請檢查 NAME, ANOMALY 和 CHECKPOINT_STEP 設定"
    exit 1
fi

# 檢查基礎模型是否存在
if [ ! -d "$MODEL_NAME" ]; then
    echo "❌ 錯誤: 基礎模型不存在: $MODEL_NAME"
    echo "請先下載或檢查模型路徑"
    exit 1
fi

# 執行視覺化
echo ""
echo "🚀 開始視覺化 attention maps..."
echo ""

.env/bin/python inference/visualize_attention.py \
    --model_name "$MODEL_NAME" \
    --lora_weights "$LORA_WEIGHTS" \
    --prompt "$PROMPT" \
    --num_inference_steps $NUM_INFERENCE_STEPS \
    --output_dir "$OUTPUT_DIR" \
    --attention_res $ATTENTION_RES \
    --seed $SEED \
    --guidance_scale $GUIDANCE_SCALE

# 檢查執行結果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✅ 完成！Attention maps 已儲存至:"
    echo "   $OUTPUT_DIR"
    echo "================================================"
    echo ""
    echo "📂 輸出檔案:"
    ls -lh "$OUTPUT_DIR"
else
    echo ""
    echo "❌ 執行失敗，請檢查錯誤訊息"
    exit 1
fi
