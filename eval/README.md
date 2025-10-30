# 評估工具使用指南 (DualAnoDiff 方法)

本目錄包含從 DualAnoDiff 專案遷移過來的準確評估工具，用於評估生成圖片的質量和異常定位性能。

## 📁 目錄結構

```
eval/
├── compute-ic-lpips.py       # IC-LPIPS 評估（類內多樣性）
├── compute-is.py              # Inception Score 評估
├── train-localization.py      # 訓練異常定位模型
├── test-localization.py       # 測試異常定位性能
├── unet_utils/                # 定位模型工具包
│   ├── model_unet.py          # UNet 模型
│   ├── data_loader.py         # 數據加載器
│   ├── au_pro_util.py         # PRO 指標計算
│   └── ...
├── compute_ic_lpips.py        # 舊版 IC-LPIPS（已備份）
└── compute_is.py              # 舊版 IS（已備份）
```

## 🚀 快速開始

### 1️⃣ 圖像生成質量評估

#### 評估單個樣本的所有指標
```bash
# 在專案根目錄執行
bash script/eval/eval_all.sh hazelnut
```

#### 單獨評估 IC-LPIPS（類內多樣性）
```bash
bash script/eval/eval_ic_lpips.sh hazelnut
```

#### 單獨評估 Inception Score
```bash
bash script/eval/eval_inception_score.sh hazelnut
```

#### 評估所有樣本
```bash
bash script/eval/eval_all.sh all
```

**輸出文件:**
- `ic_lpips_results.csv` - IC-LPIPS 結果
- `IS_results.csv` - Inception Score 結果

---

### 2️⃣ 異常定位性能評估

#### 完整流程（訓練 + 測試）
```bash
bash script/eval/localization_pipeline.sh hazelnut
```

#### 僅訓練定位模型
```bash
bash script/eval/train_localization.sh hazelnut
```

#### 僅測試定位模型
```bash
bash script/eval/test_localization.sh hazelnut
```

**輸出文件:**
- `checkpoints/localization/` - 訓練的模型
- `result.csv` - 測試結果（AUROC, AP, F1-max, PRO）
- `result/` - 視覺化結果

---

## 📊 評估指標說明

### 圖像生成質量指標

#### IC-LPIPS (Intra-class LPIPS Diversity)
- **範圍**: 0.0 ~ 1.0（越高越好）
- **意義**: 衡量生成圖片的多樣性
- **計算方法**:
  1. 將生成圖片與原始圖片配對分群
  2. 計算每個群內所有圖片對的 LPIPS 距離
  3. 對所有群取平均

#### Inception Score (IS)
- **範圍**: 1.0 ~ ∞（越高越好）
- **意義**: 衡量生成圖片的質量和多樣性
- **計算方法**: 使用 Inception-v3 模型計算 KL 散度
- **使用工具**: torch-fidelity（標準實現）

### 異常定位指標

#### Image-level (圖像級別)
- **AUROC-I**: Area Under ROC Curve（越高越好）
- **AP-I**: Average Precision（越高越好）
- **F1-max-I**: 最大 F1 分數（越高越好）

#### Pixel-level (像素級別)
- **AUROC-P**: 像素級 AUROC（越高越好）
- **AP-P**: 像素級 AP（越高越好）
- **F1-max-P**: 像素級最大 F1 分數（越高越好）
- **PRO-P**: Per-Region Overlap（越高越好）

---

## 🔧 高級用法

### 自定義路徑

#### IC-LPIPS 評估
```bash
.env/bin/python eval/compute-ic-lpips.py \
    --sample_name hazelnut \
    --generate_data_path generate_data \
    --mvtec_path datasets/mvtec_ad \
    --output my_ic_lpips_results.csv
```

#### Inception Score 評估
```bash
.env/bin/python eval/compute-is.py \
    --sample_name hazelnut \
    --generate_data_path generate_data \
    --output my_IS_results.csv \
    --gpu 0
```

#### 訓練定位模型（自定義參數）
```bash
.env/bin/python eval/train-localization.py \
    --sample_name hazelnut \
    --generated_data_path generate_data \
    --mvtec_path datasets/mvtec_ad \
    --save_path checkpoints/my_localization \
    --bs 32 \
    --lr 0.0002 \
    --epochs 300 \
    --gpu_id 0
```

#### 測試定位模型
```bash
.env/bin/python eval/test-localization.py \
    --sample_name hazelnut \
    --mvtec_path datasets/mvtec_ad \
    --checkpoint_path checkpoints/my_localization \
    --gpu_id 0
```

---

## 📋 數據格式要求

### 生成數據目錄結構
```
generate_data/
└── hazelnut/
    ├── crack/
    │   └── image/
    │       ├── 0.png
    │       ├── 1.png
    │       └── ...
    ├── cut/
    │   └── image/
    ├── hole/
    │   └── image/
    └── print/
        └── image/
```

### MVTec 數據集結構
```
datasets/mvtec_ad/
└── hazelnut/
    └── test/
        ├── crack/
        │   ├── 000.png
        │   ├── 000_mask.png
        │   └── ...
        ├── cut/
        ├── hole/
        └── print/
```

---

## 🆚 與舊版評估方法的差異

### DualAnoDiff 方法（新，更準確）
✅ 使用 torch-fidelity 計算 IS（標準實現）
✅ IC-LPIPS 先與原始圖片分群（更符合論文）
✅ 支持按樣本名稱評估多個缺陷類型
✅ 自動計算平均分數
✅ 整合異常定位評估工具

### TengTengDiff 舊方法
❌ 手動實現 IS 計算
❌ IC-LPIPS 隨機分群
❌ 需要手動指定每個圖片目錄

---

## 🔍 故障排除

### 常見錯誤

#### 1. 找不到 lpips 或 torch-fidelity
```bash
conda run -p .env pip install lpips torch-fidelity
```

#### 2. CUDA out of memory
- 減少 batch size: `--bs 8`
- 使用較小的樣本量

#### 3. checkpoint 文件不存在
- 先運行 `train_localization.sh` 訓練模型
- 檢查 `checkpoints/localization/` 目錄

---

## 📚 參考文獻

- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
- **Inception Score**: Salimans et al., "Improved Techniques for Training GANs"
- **IC-LPIPS**: Ojha et al., "Few-shot Image Generation via Cross-domain Correspondence"
- **PRO**: Bergmann et al., "Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection"

---

## 📞 聯絡資訊

如有問題或建議，請參考：
- DualAnoDiff 專案: https://github.com/yinyjin/DualAnoDiff
- 本專案的評估工具文檔: `/home/bluestar/research/TengTengDiff/eval/README.md`

---

**更新日期**: 2025-10-27
**版本**: DualAnoDiff Evaluation Tools v1.0
