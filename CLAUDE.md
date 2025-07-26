# CLAUDE.md

此文件為 Claude Code (claude.ai/code) 在此專案中工作時提供指引。

## 專案概述

TengTengDiff 是一個基於 DreamBooth LoRA 的訓練和推論系統，專門用於 MVTec-AD 資料集的異常檢測。它使用 Stable Diffusion v1.5 配合 LoRA 適配器來生成合成的異常資料。

核心概念是使用 DreamBooth 技術微調 Stable Diffusion，讓它學會生成特定類型的工業缺陷圖片，用於擴充異常檢測的訓練資料。

## 主要指令

### 訓練
```bash
bash script/train.sh
```
訓練特定 MVTec-AD 異常類型的 LoRA 模型。主要參數：
- `NAME`：MVTec 類別（如 "hazelnut"）
- `ANOMALY`：異常類型（如 "hole"）
- 預設：1000 訓練步數、批次大小 4、學習率 2e-5

### 推論
```bash
bash script/inference.sh
```
使用訓練好的 LoRA 權重生成合成異常圖片。

### 執行分割（U-2-Net）
```bash
bash script/segmentor.sh
```
使用 U-2-Net 模型進行分割任務。

## 架構說明

### 核心元件

1. **訓練管線** (`train_dreambooth_lora.py`)
   - Stable Diffusion 的 DreamBooth LoRA 微調
   - 支援文字編碼器訓練
   - MVTec-AD 專用資料載入
   - 使用 accelerate 進行分散式訓練

2. **推論管線** (`inference.py`, `inference2.py`)
   - 載入基礎 SD 模型 + LoRA 權重
   - 使用提示詞生成異常圖片
   - 輸出至 `generate_data/` 目錄

3. **U-2-Net 整合** (`U-2-Net/`)
   - 顯著物體檢測模型
   - 用於分割任務
   - 模型檢查點：`models/u2net.pth`

4. **Diffusers 函式庫** (`diffusers/`)
   - HuggingFace diffusers 的本地副本
   - 包含所有管線元件
   - 針對專案需求的客製化修改

### 資料結構

- **MVTec-AD 資料集**：`datasets/mvtec_ad/`
  - 類別：bottle、cable、capsule、carpet、grid、hazelnut、leather、metal_nut、pill、screw、tile、toothbrush、transistor、wood、zipper
  - 每個類別都有異常子類型

- **模型權重**：
  - 基礎模型：`models/stable-diffusion-v1-5/`
  - LoRA 檢查點：`all_generate/{類別}/{異常}/checkpoint-{步數}/`
  - U-2-Net：`models/u2net.pth`

- **輸出**：
  - 生成圖片：`generate_data/{類別}/{異常}/`
  - 訓練日誌：TensorBoard 格式

### 關鍵參數

- **訓練**：
  - 解析度：512x512
  - 混合精度：停用（為了穩定性）
  - LoRA rank：32
  - 提示詞："a vfx with sks"（混合）、"sks"（前景）

- **推論**：
  - 預設：100 推論步數
  - 排程器：DPMSolverMultistep

## 環境設定

### Conda 環境
此專案使用 conda 環境，位於 `.env` 目錄：
- Python 版本：3.12.11
- PyTorch 版本：2.7.1+cu126
- Diffusers 版本：0.35.0.dev0

### 環境使用方式

直接使用環境中的 Python：
```bash
# 執行腳本
.env/bin/python script.py

# 安裝套件
.env/bin/pip install package_name

# 執行訓練
.env/bin/python train_dreambooth_lora.py [參數...]

# 執行推論
.env/bin/python inference.py [參數...]
```

## 開發提示

- 監控 GPU 記憶體使用量 - 批次大小 4 需要約 16GB VRAM
- 查看 TensorBoard 日誌以追蹤訓練進度
- 使用 `accelerate launch` 進行多 GPU 設定
- 如果遇到 CUDA 相關問題，確認 PyTorch 的 CUDA 版本（目前為 12.6）與系統相符