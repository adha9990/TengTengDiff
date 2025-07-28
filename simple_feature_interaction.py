import torch
import torch.nn.functional as F


def compute_feature_interaction_loss(model_input_blend, model_input_fg, noise_blend, noise_fg):
    """
    計算特徵交互損失，鼓勵背景和異常特徵的互補學習
    
    Args:
        model_input_blend: 混合圖像的潛在表示
        model_input_fg: 前景（異常）的潛在表示
        noise_blend: 混合圖像的噪聲
        noise_fg: 前景的噪聲
    
    Returns:
        interaction_loss: 特徵交互損失
    """
    
    # 1. 背景一致性損失 - 確保混合圖像保留背景信息
    # 使用低頻分量比較（背景通常是低頻）
    blend_low_freq = F.avg_pool2d(model_input_blend, kernel_size=4, stride=4)
    fg_low_freq = F.avg_pool2d(model_input_fg, kernel_size=4, stride=4)
    
    # 背景應該在 blend 和 fg 之間有所不同（fg 主要是異常）
    bg_consistency_loss = -F.mse_loss(blend_low_freq, fg_low_freq)
    
    # 2. 形狀保持損失 - 確保異常區域的形狀信息被保留
    # 使用高頻分量（邊緣和細節）
    blend_high_freq = model_input_blend - F.interpolate(blend_low_freq, size=model_input_blend.shape[-2:], mode='bilinear', align_corners=False)
    fg_high_freq = model_input_fg - F.interpolate(fg_low_freq, size=model_input_fg.shape[-2:], mode='bilinear', align_corners=False)
    
    # 高頻分量應該相關（異常的形狀特徵）
    shape_correlation = F.cosine_similarity(
        blend_high_freq.flatten(1), 
        fg_high_freq.flatten(1), 
        dim=1
    ).mean()
    
    # 3. 互補性損失 - 鼓勵 blend 和 fg 學習不同的特徵
    # 計算特徵的正交性
    blend_flat = model_input_blend.flatten(1)
    fg_flat = model_input_fg.flatten(1)
    
    # 歸一化
    blend_norm = F.normalize(blend_flat, p=2, dim=1)
    fg_norm = F.normalize(fg_flat, p=2, dim=1)
    
    # 計算餘弦相似度（越小越正交）
    orthogonality_loss = torch.abs(F.cosine_similarity(blend_norm, fg_norm, dim=1).mean())
    
    # 4. 噪聲相關性損失 - 確保噪聲模式有所不同
    noise_correlation = F.cosine_similarity(
        noise_blend.flatten(1),
        noise_fg.flatten(1),
        dim=1
    ).mean()
    
    # 總交互損失
    interaction_loss = (
        0.1 * torch.abs(bg_consistency_loss) +  # 背景差異
        0.3 * (1 - shape_correlation) +          # 形狀相關性
        0.3 * orthogonality_loss +               # 特徵正交性
        0.1 * torch.abs(noise_correlation)       # 噪聲差異
    )
    
    return interaction_loss


def compute_mask_alignment_loss(model_pred_blend, model_pred_fg):
    """
    計算遮罩對齊損失，確保異常區域的一致性
    
    Args:
        model_pred_blend: 混合圖像的預測
        model_pred_fg: 前景的預測
    
    Returns:
        alignment_loss: 對齊損失
    """
    
    # 計算預測的差異（異常區域應該有較大差異）
    diff = torch.abs(model_pred_blend - model_pred_fg)
    
    # 生成軟遮罩（差異大的地方是異常區域）
    soft_mask = torch.sigmoid(10 * (diff.mean(dim=1, keepdim=True) - 0.1))
    
    # 對齊損失 - 在異常區域，fg 應該有更強的響應
    fg_magnitude = torch.abs(model_pred_fg).mean(dim=1, keepdim=True)
    blend_magnitude = torch.abs(model_pred_blend).mean(dim=1, keepdim=True)
    
    # 在異常區域（soft_mask 高的地方），fg 應該比 blend 有更強的響應
    alignment_loss = F.mse_loss(
        soft_mask * fg_magnitude,
        soft_mask * torch.maximum(blend_magnitude, fg_magnitude)
    )
    
    return alignment_loss