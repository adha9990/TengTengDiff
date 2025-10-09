#!/usr/bin/env python3
"""
視覺化 Diffusion Model 的 Cross-Attention Maps
使用專案內建的 AttentionStore 功能
"""

import argparse
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from utils.ptp_utils import AttentionStore, register_attention_control, aggregate_attention


def parse_args():
    parser = argparse.ArgumentParser(description="視覺化 cross-attention maps")
    parser.add_argument(
        "--model_name",
        type=str,
        default="models/stable-diffusion-v1-5",
        help="基礎 SD 模型路徑"
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="LoRA 權重路徑（例如：all_generate/hazelnut/full/checkpoint-5000）"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a vfx",
        help="生成提示詞"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="推論步數"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attention_maps",
        help="輸出目錄"
    )
    parser.add_argument(
        "--attention_res",
        type=int,
        default=16,
        help="Attention map 解析度（16 或 32）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )

    return parser.parse_args()


def overlay_attention_on_image(image, attention_map, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    將 attention map 疊加在圖片上

    Args:
        image: PIL Image 或 numpy array
        attention_map: 2D attention map (numpy array)
        alpha: 疊加透明度
        colormap: OpenCV colormap

    Returns:
        疊加後的 PIL Image
    """
    # 轉換圖片為 numpy array
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    # 確保圖片是 RGB
    if img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[:, :, :3]

    # 將 attention map 調整到圖片大小
    h, w = img_array.shape[:2]
    attention_resized = cv2.resize(attention_map, (w, h), interpolation=cv2.INTER_CUBIC)

    # 正規化 attention map 到 0-255
    attention_norm = (attention_resized - attention_resized.min()) / (attention_resized.max() - attention_resized.min() + 1e-8)
    attention_norm = (attention_norm * 255).astype(np.uint8)

    # 應用 colormap
    attention_colored = cv2.applyColorMap(attention_norm, colormap)
    attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)

    # 疊加
    overlay = cv2.addWeighted(img_array, 1 - alpha, attention_colored, alpha, 0)

    return Image.fromarray(overlay)


def create_comparison_view(image, attention_maps, tokens, output_path, top_k=6):
    """
    創建對比視圖：原圖 + top-k tokens 的 attention maps

    Args:
        image: 生成的圖片
        attention_maps: attention map 陣列 [H, W, num_tokens]
        tokens: token 列表
        output_path: 輸出路徑
        top_k: 顯示前 k 個重要的 tokens
    """
    # 計算每個 token 的平均 attention
    token_importance = attention_maps.mean(axis=(0, 1))

    # 找出 top-k tokens（排除 start/end tokens）
    valid_indices = []
    for idx in range(len(tokens)):
        token = tokens[idx]
        if token not in ['<|startoftext|>', '<|endoftext|>'] and not token.startswith('<pad_'):
            valid_indices.append(idx)

    # 如果沒有有效 tokens，使用所有非 padding tokens
    if not valid_indices:
        valid_indices = [i for i in range(min(len(tokens), attention_maps.shape[2]))]

    # 按重要性排序
    valid_importance = [(idx, token_importance[idx]) for idx in valid_indices if idx < len(token_importance)]
    valid_importance.sort(key=lambda x: x[1], reverse=True)

    # 選擇 top-k
    top_indices = [idx for idx, _ in valid_importance[:top_k]]

    # 如果不足 top_k，添加最重要的 tokens（包括 special tokens）
    if len(top_indices) < top_k:
        all_indices_sorted = np.argsort(token_importance)[::-1]
        for idx in all_indices_sorted:
            if idx not in top_indices and len(top_indices) < top_k:
                top_indices.append(idx)

    # 創建視覺化
    num_plots = len(top_indices) + 1  # +1 for original image
    num_cols = min(4, num_plots)
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 4, num_rows * 4))
    axes = axes.flatten() if num_plots > 1 else [axes]

    # 顯示原圖
    axes[0].imshow(image)
    axes[0].set_title("Generated Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 顯示 top-k tokens 的 attention maps
    for plot_idx, token_idx in enumerate(top_indices, start=1):
        if plot_idx >= len(axes):
            break

        token_label = tokens[token_idx] if token_idx < len(tokens) else f"<pad_{token_idx}>"
        attn_map = attention_maps[:, :, token_idx]
        importance = token_importance[token_idx]

        # 疊加 attention 在圖片上
        overlay = overlay_attention_on_image(image, attn_map, alpha=0.5)

        axes[plot_idx].imshow(overlay)
        axes[plot_idx].set_title(
            f"Token {token_idx}: '{token_label}'\nImportance: {importance:.4f}",
            fontsize=10
        )
        axes[plot_idx].axis('off')

    # 隱藏多餘的子圖
    for idx in range(num_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 對比視圖已儲存至: {output_path}")


def create_aggregated_heatmap(image, attention_maps, tokens, output_path):
    """
    創建聚合所有內容 tokens 的熱力圖

    Args:
        image: 生成的圖片
        attention_maps: attention map 陣列 [H, W, num_tokens]
        tokens: token 列表
        output_path: 輸出路徑
    """
    # 找出內容 tokens（排除 special tokens 和 padding）
    content_indices = []
    for idx in range(min(len(tokens), attention_maps.shape[2])):
        token = tokens[idx] if idx < len(tokens) else f"<pad_{idx}>"
        if token not in ['<|startoftext|>', '<|endoftext|>'] and not token.startswith('<pad_'):
            content_indices.append(idx)

    if not content_indices:
        print("⚠️  警告: 沒有找到內容 tokens，使用所有 tokens")
        content_indices = list(range(min(len(tokens), attention_maps.shape[2])))

    # 聚合內容 tokens 的 attention
    aggregated = attention_maps[:, :, content_indices].sum(axis=2)

    # 創建視覺化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原圖
    axes[0].imshow(image)
    axes[0].set_title("Generated Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 純熱力圖
    im = axes[1].imshow(aggregated, cmap='jet', interpolation='bilinear')
    axes[1].set_title("Content Tokens Attention\n(Aggregated)", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 疊加視圖
    overlay = overlay_attention_on_image(image, aggregated, alpha=0.6)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay on Image", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # 添加 tokens 資訊
    content_tokens = [tokens[i] if i < len(tokens) else f"<pad_{i}>" for i in content_indices]
    fig.suptitle(f"Aggregated tokens: {', '.join(content_tokens)}", fontsize=10, y=0.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 聚合熱力圖已儲存至: {output_path}")


def visualize_cross_attention(
    attention_maps,
    tokens,
    output_path,
    res=16,
    from_where=["up", "down", "mid"]
):
    """
    視覺化 cross-attention maps

    Args:
        attention_maps: AttentionStore 物件
        tokens: 提示詞的 token 列表
        output_path: 輸出圖片路徑
        res: attention map 解析度
        from_where: 要聚合的 UNet 位置
    """
    # 聚合 attention maps
    try:
        attention = aggregate_attention(
            attention_store=attention_maps,
            res=res,
            from_where=from_where,
            is_cross=True,
            select=0  # 選擇第一個樣本
        )
    except RuntimeError as e:
        if "expected a non-empty list" in str(e):
            print(f"⚠️  警告: 在 {from_where} 層找不到解析度 {res}x{res} 的 attention maps，跳過此視覺化")
            return
        else:
            raise

    # attention shape: [res, res, num_tokens]
    total_tokens = attention.shape[-1]

    # 只顯示實際的 prompt tokens（不包括 padding）
    actual_token_count = min(len(tokens), total_tokens)
    display_tokens = actual_token_count  # 只顯示實際 tokens

    # 創建視覺化網格
    num_cols = min(8, display_tokens)
    num_rows = (display_tokens + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    if display_tokens == 1:
        axes = np.array([axes])
    elif num_rows == 1:
        axes = axes.reshape(1, -1)

    axes = axes.flatten()

    for idx in range(display_tokens):
        ax = axes[idx]

        # 獲取當前 token 的 attention map
        attn_map = attention[:, :, idx].cpu().numpy()

        # 顯示 attention map
        im = ax.imshow(attn_map, cmap='jet', interpolation='bilinear')

        # 處理 token 標籤
        token_label = tokens[idx] if idx < len(tokens) else f"<pad_{idx}>"

        # 計算此 token 的 attention 強度
        avg_attention = attn_map.mean()

        ax.set_title(f"Token {idx}: '{token_label}'\nAvg: {avg_attention:.4f}", fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 隱藏多餘的子圖
    for idx in range(display_tokens, len(axes)):
        axes[idx].axis('off')

    # 添加整體標題
    location_str = '+'.join(from_where)
    fig.suptitle(f"Cross-Attention Maps ({location_str} layers, {res}x{res})", fontsize=14, y=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Attention map 已儲存至: {output_path} (顯示 {display_tokens}/{total_tokens} tokens)")


def save_attention_summary(
    attention_maps,
    tokens,
    output_path,
    res=16,
    from_where=["up", "down", "mid"]
):
    """
    儲存每個 token 的平均 attention 強度摘要
    """
    try:
        attention = aggregate_attention(
            attention_store=attention_maps,
            res=res,
            from_where=from_where,
            is_cross=True,
            select=0
        )
    except RuntimeError as e:
        if "expected a non-empty list" in str(e):
            print(f"⚠️  警告: 在 {from_where} 層找不到解析度 {res}x{res} 的 attention maps，跳過摘要生成")
            return
        else:
            raise

    # 計算每個 token 的平均 attention
    token_attention = attention.mean(dim=[0, 1]).cpu().numpy()

    # 儲存摘要
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Token Attention Summary\n")
        f.write("=" * 50 + "\n\n")

        # 處理 token 數量可能不匹配的情況
        num_attn_tokens = len(token_attention)
        for idx in range(num_attn_tokens):
            token_label = tokens[idx] if idx < len(tokens) else f"<pad_{idx}>"
            attn_value = token_attention[idx]
            f.write(f"Token {idx:2d} | {token_label:15s} | Attention: {attn_value:.6f}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Total tokens (prompt): {len(tokens)}\n")
        f.write(f"Total tokens (attention): {num_attn_tokens}\n")
        f.write(f"Resolution: {res}x{res}\n")
        f.write(f"Aggregated from: {', '.join(from_where)}\n")

    print(f"✅ Attention 摘要已儲存至: {output_path}")


def main():
    args = parse_args()

    # 設定隨機種子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 創建輸出目錄
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Cross-Attention Map 視覺化工具")
    print("=" * 60)
    print(f"模型: {args.model_name}")
    print(f"LoRA: {args.lora_weights}")
    print(f"提示詞: {args.prompt}")
    print(f"推論步數: {args.num_inference_steps}")
    print(f"Attention 解析度: {args.attention_res}x{args.attention_res}")
    print("=" * 60 + "\n")

    # 載入 pipeline
    print("📥 載入 Stable Diffusion pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )

    # 設定 scheduler
    scheduler_args = {}
    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type
        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"
        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, **scheduler_args
    )

    # 載入 LoRA 權重
    print(f"📥 載入 LoRA 權重: {args.lora_weights}")
    pipe.load_lora_weights(args.lora_weights)

    # 移至 GPU
    pipe = pipe.to("cuda")

    # 創建 AttentionStore 並註冊
    print("🔧 註冊 attention control...")
    attention_store = AttentionStore(save_global_store=True)
    register_attention_control(pipe, attention_store)

    # Tokenize 提示詞以獲取 token 資訊
    tokens = pipe.tokenizer.encode(args.prompt)
    token_strings = [pipe.tokenizer.decode([token]) for token in tokens]

    print(f"\n📝 提示詞 tokens ({len(token_strings)} 個):")
    for idx, token in enumerate(token_strings):
        print(f"  {idx:2d}: {token}")
    print()

    # 生成圖片
    print("🎨 生成圖片並收集 attention maps...")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator
    ).images[0]

    # 儲存生成的圖片
    output_image_path = os.path.join(args.output_dir, "generated_image.png")
    result.save(output_image_path)
    print(f"✅ 生成圖片已儲存至: {output_image_path}")

    # 視覺化 cross-attention maps
    print("\n📊 視覺化 cross-attention maps...")

    # 1. 創建聚合熱力圖（最重要的視覺化）
    print("\n🔥 生成聚合熱力圖...")
    try:
        attention_all = aggregate_attention(
            attention_store=attention_store,
            res=args.attention_res,
            from_where=["up", "down"],  # 使用 up + down，mid 可能沒有對應解析度
            is_cross=True,
            select=0
        )
        attention_np = attention_all.cpu().numpy()

        aggregated_path = os.path.join(args.output_dir, "attention_aggregated.png")
        create_aggregated_heatmap(result, attention_np, token_strings, aggregated_path)
    except RuntimeError as e:
        print(f"⚠️  警告: 無法生成聚合熱力圖: {e}")

    # 2. 創建 top-k tokens 對比視圖
    print("\n📊 生成 Top-K Tokens 對比視圖...")
    try:
        attention_all = aggregate_attention(
            attention_store=attention_store,
            res=args.attention_res,
            from_where=["up", "down"],
            is_cross=True,
            select=0
        )
        attention_np = attention_all.cpu().numpy()

        comparison_path = os.path.join(args.output_dir, "attention_comparison.png")
        create_comparison_view(result, attention_np, token_strings, comparison_path, top_k=6)
    except RuntimeError as e:
        print(f"⚠️  警告: 無法生成對比視圖: {e}")

    # 3. 為不同的 UNet 位置創建詳細視覺化
    print("\n🗺️  生成各層詳細 attention maps...")
    locations = [
        (["up", "down"], "all"),  # 合併 up 和 down
        (["down"], "down"),
        (["up"], "up"),
    ]

    for from_where, name in locations:
        output_path = os.path.join(args.output_dir, f"attention_detail_{name}.png")
        visualize_cross_attention(
            attention_maps=attention_store,
            tokens=token_strings,
            output_path=output_path,
            res=args.attention_res,
            from_where=from_where
        )

    # 4. 儲存 attention 摘要
    print("\n📝 生成 attention 摘要...")
    summary_path = os.path.join(args.output_dir, "attention_summary.txt")
    save_attention_summary(
        attention_maps=attention_store,
        tokens=token_strings,
        output_path=summary_path,
        res=args.attention_res,
        from_where=["up", "down"]
    )

    print("\n" + "=" * 60)
    print("✨ 完成！所有檔案已儲存至:", args.output_dir)
    print("=" * 60)

    # 列出輸出檔案
    print("\n📂 輸出檔案:")
    for filename in sorted(os.listdir(args.output_dir)):
        filepath = os.path.join(args.output_dir, filename)
        size = os.path.getsize(filepath)
        print(f"  • {filename} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
