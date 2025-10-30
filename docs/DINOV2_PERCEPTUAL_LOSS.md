# DINOv2 Perceptual Loss Integration

## 概述

本專案已整合 DINOv2 特徵感知損失，用於提升 DreamBooth LoRA 訓練的圖像品質。DINOv2 是 Meta AI 開發的強大視覺 Transformer 模型，能夠提取豐富的語義特徵，指導圖像生成保持感知品質。

## 功能特點

- ✅ 支援多層特徵提取（預設使用層 3, 6, 9, 11）
- ✅ 可配置的損失權重和損失類型（L1, L2, Smooth L1）
- ✅ 支援多種 DINOv2 模型變體（ViT-S, ViT-B, ViT-L, ViT-G）
- ✅ 自動處理潛在空間到像素空間的轉換
- ✅ 同時支援 Stage 1 和 Stage 2 訓練

## 使用方法

### Stage 1 訓練

在 Stage 1 訓練中啟用 DINOv2 感知損失：

```bash
bash script/train/train_stage1-full.sh
```

訓練腳本已包含以下 DINOv2 參數：
```bash
--use_dinov2_loss \                    # 啟用 DINOv2 感知損失
--dinov2_loss_weight=0.1 \             # 感知損失權重
--dinov2_model_name="vitb14" \         # DINOv2 模型變體
--dinov2_loss_type="l2" \              # 損失類型
--dinov2_feature_layers 3 6 9 11       # 提取特徵的層
```

### Stage 2 訓練

在 Stage 2 訓練中啟用 DINOv2 感知損失：

```bash
bash script/train/train_stage2-dual.sh
```

同樣包含相同的 DINOv2 參數配置。

### 自訂配置

#### 1. 損失權重 (--dinov2_loss_weight)

控制感知損失相對於主要 MSE 損失的權重：

- **0.05-0.1**：推薦值，平衡 MSE 和感知損失
- **0.1-0.3**：更重視感知相似度
- **0.3+**：可能過度強調感知特徵

```bash
--dinov2_loss_weight=0.1    # 預設：0.1
```

#### 2. 模型變體 (--dinov2_model_name)

選擇不同大小的 DINOv2 模型：

| 模型 | 參數量 | GPU 記憶體 | 特徵品質 | 推薦場景 |
|------|--------|-----------|---------|---------|
| vits14 | 21M | 低 (~1GB) | 良好 | 記憶體受限 |
| **vitb14** | 86M | 中等 (~2GB) | **優秀（推薦）** | **一般使用** |
| vitl14 | 304M | 高 (~4GB) | 非常好 | 高品質需求 |
| vitg14 | 1.1B | 很高 (~8GB) | 最佳 | 極致品質 |

```bash
--dinov2_model_name="vitb14"    # 預設：vitb14
```

#### 3. 損失類型 (--dinov2_loss_type)

選擇特徵比較的損失函數：

- **l2**（MSE）：預設，平滑且穩定
- **l1**（MAE）：對離群值更魯棒
- **smooth_l1**：結合 L1 和 L2 的優點

```bash
--dinov2_loss_type="l2"    # 預設：l2
```

#### 4. 特徵層 (--dinov2_feature_layers)

指定從哪些 Transformer 層提取特徵：

- **淺層（0-3）**：捕捉低層次特徵（邊緣、紋理）
- **中層（4-8）**：捕捉中層次特徵（局部結構）
- **深層（9-11）**：捕捉高層次特徵（語義信息）

```bash
--dinov2_feature_layers 3 6 9 11    # 預設：多層組合
```

## 訓練範例

### 完整 Stage 1 訓練命令

```bash
export MODEL_NAME="models/stable-diffusion-v1-5"
export INSTANCE_DIR="datasets/mvtec_ad"
export NAME="hazelnut"
export OUTPUT_DIR="all_generate/"

accelerate launch train/stage1-full/train.py \
    --mixed_precision="no" \
    --mvtec_name=$NAME \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir="$OUTPUT_DIR/$NAME/stage1-full" \
    --instance_prompt="a vfx" \
    --resolution=512 \
    --train_batch_size=8 \
    --learning_rate=2e-5 \
    --max_train_steps=5000 \
    --rank 32 \
    --train_text_encoder \
    --use_dinov2_loss \
    --dinov2_loss_weight=0.1 \
    --dinov2_model_name="vitb14" \
    --dinov2_loss_type="l2" \
    --dinov2_feature_layers 3 6 9 11
```

### 完整 Stage 2 訓練命令

```bash
export NAME="hazelnut"
export ANOMALY="hole"

accelerate launch train/stage2-dual/train.py \
    --mixed_precision="no" \
    --mvtec_name=$NAME \
    --mvtec_anamaly_name=$ANOMALY \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir="$OUTPUT_DIR/$NAME/stage2-$ANOMALY-dual" \
    --instance_prompt_blend="a vfx with sks" \
    --instance_prompt_fg="sks" \
    --resolution=512 \
    --train_batch_size=4 \
    --learning_rate=2e-5 \
    --max_train_steps=8000 \
    --resume_from_checkpoint="$OUTPUT_DIR/$NAME/stage1-full/checkpoint-5000" \
    --rank 32 \
    --train_text_encoder \
    --use_dinov2_loss \
    --dinov2_loss_weight=0.1 \
    --dinov2_model_name="vitb14" \
    --dinov2_loss_type="l2" \
    --dinov2_feature_layers 3 6 9 11
```

## 技術細節

### 實現原理

1. **特徵提取**：從 DINOv2 的多個 Transformer 層提取特徵
2. **潛在空間重建**：從預測的噪聲重建乾淨的潛在表示
3. **解碼**：使用 VAE 將潛在空間解碼為像素空間
4. **損失計算**：計算預測圖像和目標圖像之間的特徵距離

### 支援的預測類型

- ✅ **epsilon prediction**（噪聲預測）：Stable Diffusion v1.5 預設
- ✅ **v-prediction**：某些 diffusion 模型使用的預測方式

### 記憶體考量

啟用 DINOv2 感知損失會增加 GPU 記憶體使用：

| 模型 | 額外記憶體 | 建議 GPU |
|------|-----------|---------|
| vits14 | ~1GB | ≥12GB |
| vitb14 | ~2GB | ≥16GB |
| vitl14 | ~4GB | ≥24GB |
| vitg14 | ~8GB | ≥40GB |

**優化建議**：
- 減小批次大小（`--train_batch_size`）
- 使用梯度累積（`--gradient_accumulation_steps`）
- 選擇較小的模型變體

## 效果評估

### 預期改進

使用 DINOv2 感知損失應該能帶來：

1. **感知品質提升**：生成的圖像更符合人類視覺感知
2. **細節保留**：更好地保留物體的細節和紋理
3. **語義一致性**：生成的異常更符合語義要求

### 評估指標

可以使用以下指標評估改進效果：

```bash
# 計算 IC-LPIPS（類內 LPIPS）
.env/bin/python eval/compute_ic_lpips.py --image_dir generate_data/hazelnut/full

# 計算 Inception Score
.env/bin/python eval/compute_is.py --image_dir generate_data/hazelnut/full
```

## 停用 DINOv2 損失

如果想要停用 DINOv2 感知損失，只需移除 `--use_dinov2_loss` 參數：

```bash
# 從訓練腳本中移除以下行
--use_dinov2_loss \
--dinov2_loss_weight=0.1 \
--dinov2_model_name="vitb14" \
--dinov2_loss_type="l2" \
--dinov2_feature_layers 3 6 9 11
```

或編輯訓練腳本，註釋掉相關參數。

## 故障排除

### 問題：DINOv2 模型下載失敗

**解決方案**：DINOv2 模型會自動從 torch.hub 下載。確保網路連接正常，或手動下載權重。

### 問題：GPU 記憶體不足

**解決方案**：
1. 減小批次大小：`--train_batch_size=4` → `--train_batch_size=2`
2. 使用較小的模型：`vitb14` → `vits14`
3. 減小損失權重或停用感知損失

### 問題：訓練速度變慢

**解決方案**：
- DINOv2 感知損失會增加約 20-30% 的訓練時間
- 考慮使用較小的模型變體
- 調整特徵層數量（使用更少的層）

## 相關文件

- [DINOv2 論文](https://arxiv.org/abs/2304.07193)
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2)
- [Perceptual Loss 論文](https://arxiv.org/abs/1603.08155)

## 參考配置

### 輕量級配置（記憶體受限）

```bash
--use_dinov2_loss \
--dinov2_loss_weight=0.05 \
--dinov2_model_name="vits14" \
--dinov2_loss_type="l2" \
--dinov2_feature_layers 3 9
```

### 標準配置（推薦）

```bash
--use_dinov2_loss \
--dinov2_loss_weight=0.1 \
--dinov2_model_name="vitb14" \
--dinov2_loss_type="l2" \
--dinov2_feature_layers 3 6 9 11
```

### 高品質配置（GPU 充足）

```bash
--use_dinov2_loss \
--dinov2_loss_weight=0.15 \
--dinov2_model_name="vitl14" \
--dinov2_loss_type="l2" \
--dinov2_feature_layers 3 6 9 11
```

## 更新日誌

- **2025-10-25**：初始整合 DINOv2 特徵感知損失
  - 支援 Stage 1 和 Stage 2 訓練
  - 支援多種模型變體和損失類型
  - 自動處理潛在空間轉換
