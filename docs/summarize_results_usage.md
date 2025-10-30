# summarize_results.py 使用說明

## 概述

`summarize_results.py` 已從硬編碼配置改為可配置的命令列工具，支援彈性地處理不同的評估場景。

## 主要改進

✅ **移除硬編碼** - 不再將路徑、異常類型、checkpoint 寫死在程式碼中
✅ **命令列參數** - 透過 argparse 支援完整的參數配置
✅ **向後兼容** - 保留原有預設值，現有工作流程無需修改
✅ **自動命名** - 根據目錄名稱自動生成輸出檔名

## 命令列參數

### --generate-dir
生成數據的目錄路徑

```bash
python summarize_results.py --generate-dir /path/to/generate_data/hazelnut
```

### --anomalies
異常類型列表（空格分隔）

```bash
python summarize_results.py --anomalies crack print hole cut
```

### --checkpoints
Checkpoint 步數列表（空格分隔）

```bash
python summarize_results.py --checkpoints 1000 2000 3000 4000 5000
```

### --output-name
自訂輸出檔名（選填）

```bash
python summarize_results.py --output-name my_custom_summary.csv
```

## 使用範例

### 1. 使用預設值（Hazelnut + dino.5）

```bash
.env/bin/python summarize_results.py
```

### 2. 評估不同類別（Bottle）

```bash
.env/bin/python summarize_results.py \
    --generate-dir ./generate_data_dino.5/bottle \
    --anomalies broken_large broken_small contamination
```

### 3. 評估不同 DINO 版本（dino.1）

```bash
.env/bin/python summarize_results.py \
    --generate-dir ./generate_data_dino.1/hazelnut
```

### 4. 自訂 Checkpoint 範圍

```bash
.env/bin/python summarize_results.py \
    --checkpoints 500 1500 2500 3500
```

### 5. 完整參數範例

```bash
.env/bin/python summarize_results.py \
    --generate-dir ./generate_data_dino.5/cable \
    --anomalies bent_wire cable_swap combined \
    --checkpoints 1000 2000 3000 \
    --output-name cable_results.csv
```

## 與 batch_eval_*.sh 整合

`batch_eval_hazelnut.sh` 已更新為傳遞參數給 summarize_results.py：

```bash
"${BASE_DIR}/.env/bin/python" "${BASE_DIR}/summarize_results.py" \
    --generate-dir "${GENERATE_DIR}" \
    --anomalies "${ANOMALIES[@]}" \
    --checkpoints "${CHECKPOINTS[@]}"
```

## 創建新的批量評估腳本

基於 `batch_eval_hazelnut.sh`，你可以輕鬆創建其他類別的批量評估腳本：

```bash
# 複製模板
cp script/eval/batch_eval_hazelnut.sh script/eval/batch_eval_bottle.sh

# 修改配置變數
GENERATE_DIR="${BASE_DIR}/generate_data_dino.5/bottle"
ANOMALIES=("broken_large" "broken_small" "contamination")
```

## 輸出位置

彙總報告會自動保存到：
```
{generate-dir}/{category_name}_evaluation_summary.csv
```

例如：
- `generate_data_dino.5/hazelnut/hazelnut_evaluation_summary.csv`
- `generate_data_dino.5/bottle/bottle_evaluation_summary.csv`

## 檢視說明

```bash
.env/bin/python summarize_results.py --help
```
