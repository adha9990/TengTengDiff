# CLAUDE.md

此文件為 Claude Code (claude.ai/code) 在此專案中工作時提供指引。

## 專案概述

TengTengDiff 是一個兩階段 DreamBooth LoRA 訓練和推論系統，專門用於 MVTec-AD 資料集的合成異常生成。它使用 Stable Diffusion v1.5 配合 LoRA 適配器來生成合成的工業缺陷圖片，用於擴充異常檢測的訓練資料。

系統採用兩階段方法：
- **Stage 1**：在正常圖片上訓練 LoRA，學習物體的一般外觀
- **Stage 2**：針對特定異常類型進行微調，生成缺陷樣本

## 主要指令

### Stage 1 訓練（正常圖片）
```bash
bash script/train_stage1.sh
```
在正常 MVTec-AD 圖片上訓練 LoRA。主要參數：
- `NAME`：MVTec 類別（如 "hazelnut"、"cable"、"bottle"）
- 預設：5000 步、批次大小 8、學習率 2e-5
- 輸出：`all_generate/{類別}/full/`

### Stage 2 訓練（異常特定）
```bash
bash script/train_stage2.sh
```
針對特定異常類型進行微調。主要參數：
- `NAME`：MVTec 類別
- `ANOMALY`：異常類型（如 "hole"、"crack"、"scratch"）
- 從 Stage 1 檢查點恢復
- 預設：8000 步、批次大小 4
- 輸出：`all_generate/{類別}/{異常}/`

### Stage 1 推論
```bash
bash script/inference_stage1.sh
```
使用 Stage 1 LoRA 權重生成正常合成圖片。

### Stage 2 推論
```bash
bash script/inference_stage2.sh
```
使用 Stage 2 LoRA 權重生成合成異常圖片。

### 圖片合併
```bash
bash script/merge_images.sh
```
合併生成的圖片用於視覺化或進一步處理。

### 分割（U-2-Net）
```bash
bash script/segmentor.sh
```
使用 U-2-Net 模型執行分割以生成遮罩。

## 架構說明

### 核心訓練元件

1. **兩階段訓練系統**
   - `src/stage1/`：正常圖片訓練管線
   - `src/stage2/`：異常特定微調
   - 兩階段都使用 DreamBooth LoRA 配合 Stable Diffusion v1.5
   - 支援文字編碼器訓練以提升提示詞遵循度

2. **資料流程**
   - Stage 1：從正常圖片學習一般物體外觀
   - Stage 2：從 Stage 1 檢查點開始，在異常上微調
   - 使用不同提示詞："a vfx"（Stage 1）vs "a vfx with sks"（Stage 2）

3. **模型架構**
   - 基礎：Stable Diffusion v1.5（`models/stable-diffusion-v1-5/`）
   - LoRA rank：32 用於高效適應
   - 混合精度停用以提升穩定性
   - 啟用 XFormers 以提升記憶體效率

### 資料集結構

**MVTec-AD 類別**（`datasets/mvtec_ad/`）：
- bottle、cable、capsule、carpet、grid、hazelnut、leather
- metal_nut、pill、screw、tile、toothbrush、transistor、wood、zipper

每個類別包含：
- `train/good/`：用於訓練的正常圖片
- `test/`：包含各種異常類型的測試圖片
- `ground_truth/`：異常的分割遮罩

### 輸出結構

```
all_generate/
├── {類別}/
│   ├── full/               # Stage 1 檢查點
│   │   └── checkpoint-{步數}/
│   └── {異常}/             # Stage 2 檢查點
│       └── checkpoint-{步數}/
generate_data/
└── {類別}/
    ├── full/               # Stage 1 生成的圖片
    └── {異常}/             # Stage 2 生成的異常圖片
```

## 環境設定

### Python 環境
專案使用位於 `.env/` 的 conda 環境：
- Python：3.12.11
- PyTorch：2.7.1+cu126
- Diffusers：0.35.0.dev0

### 執行指令
始終使用環境 Python：
```bash
# 執行腳本
.env/bin/python script.py

# 安裝套件
.env/bin/pip install package_name

# 直接訓練
.env/bin/python src/stage1/train.py [參數...]
.env/bin/python src/stage2/train.py [參數...]
```

## 關鍵參數

### 訓練配置
- **解析度**：512x512 像素
- **LoRA rank**：32
- **學習率**：2e-5（常數排程器）
- **Stage 1**：5000 步、批次大小 8
- **Stage 2**：8000 步、批次大小 4
- **驗證**：每個檢查點 4 張圖片
- **推論步數**：25（訓練）、50（推論）

### 提示詞
- **Stage 1**："a vfx" - 學習一般物體外觀
- **Stage 2 混合**："a vfx with sks" - 結合正常和異常
- **Stage 2 前景**："sks" - 前景異常標記

## 評估指標

### 可用評估
```bash
# 計算 IC-LPIPS（類內 LPIPS）
.env/bin/python eval/compute_ic_lpips.py

# 計算 Inception Score
.env/bin/python eval/compute_is.py
```

## 開發注意事項

- **GPU 記憶體**：批次大小 4 需要 ~16GB VRAM，批次大小 8 需要 ~24GB
- **多 GPU**：使用 `accelerate launch` 進行分散式訓練
- **監控**：TensorBoard 日誌位於 `all_generate/{類別}/{異常}/logs/`
- **CUDA 版本**：確保系統 CUDA 與 PyTorch 相符（目前為 12.6）
- **檢查點**：模型在指定間隔自動儲存
- **恢復訓練**：Stage 2 自動從 Stage 1 檢查點恢復