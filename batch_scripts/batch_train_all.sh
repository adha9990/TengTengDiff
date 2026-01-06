#!/bin/bash

# 批次執行所有訓練腳本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


echo "========================================"
echo "開始訓練: bottle"
echo "========================================"
bash "$SCRIPT_DIR/train_bottle.sh"


echo "========================================"
echo "開始訓練: cable"
echo "========================================"
bash "$SCRIPT_DIR/train_cable.sh"


echo "========================================"
echo "開始訓練: capsule"
echo "========================================"
bash "$SCRIPT_DIR/train_capsule.sh"


echo "========================================"
echo "開始訓練: carpet"
echo "========================================"
bash "$SCRIPT_DIR/train_carpet.sh"


echo "========================================"
echo "開始訓練: grid"
echo "========================================"
bash "$SCRIPT_DIR/train_grid.sh"


echo "========================================"
echo "開始訓練: leather"
echo "========================================"
bash "$SCRIPT_DIR/train_leather.sh"


echo "========================================"
echo "開始訓練: metal_nut"
echo "========================================"
bash "$SCRIPT_DIR/train_metal_nut.sh"


echo "========================================"
echo "開始訓練: pill"
echo "========================================"
bash "$SCRIPT_DIR/train_pill.sh"


echo "========================================"
echo "開始訓練: screw"
echo "========================================"
bash "$SCRIPT_DIR/train_screw.sh"


echo "========================================"
echo "開始訓練: tile"
echo "========================================"
bash "$SCRIPT_DIR/train_tile.sh"


echo "========================================"
echo "開始訓練: toothbrush"
echo "========================================"
bash "$SCRIPT_DIR/train_toothbrush.sh"


echo "========================================"
echo "開始訓練: transistor"
echo "========================================"
bash "$SCRIPT_DIR/train_transistor.sh"


echo "========================================"
echo "開始訓練: wood"
echo "========================================"
bash "$SCRIPT_DIR/train_wood.sh"


echo "========================================"
echo "開始訓練: zipper"
echo "========================================"
bash "$SCRIPT_DIR/train_zipper.sh"


echo "========================================"
echo "所有訓練完成"
echo "========================================"
