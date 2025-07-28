import torch
import torch.nn as nn
import torch.nn.functional as F


class DualFeatureInteraction(nn.Module):
    """
    實現 DualAnoDiff 的特徵交互機制
    用於在訓練過程中讓正常特徵和異常特徵相互學習
    """
    
    def __init__(self, feature_dim=1280, hidden_dim=640):
        super().__init__()
        
        # 背景特徵提取器
        self.background_extractor = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        
        # 形狀特徵提取器
        self.shape_extractor = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        
        # 交互注意力機制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            batch_first=True
        )
        
        # 特徵融合層
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 64, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
        )
        
        # 對齊層
        self.alignment_layer = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 4, 1),
        )
    
    def extract_background_features(self, latent_blend):
        """提取背景特徵"""
        return self.background_extractor(latent_blend)
    
    def extract_shape_features(self, latent_fg):
        """提取形狀特徵（異常區域）"""
        return self.shape_extractor(latent_fg)
    
    def feature_interaction(self, bg_features, fg_features):
        """特徵交互學習"""
        b, c, h, w = bg_features.shape
        
        # 展平特徵用於注意力計算
        bg_flat = bg_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        fg_flat = fg_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 交叉注意力
        attended_fg, _ = self.cross_attention(fg_flat, bg_flat, bg_flat)
        attended_bg, _ = self.cross_attention(bg_flat, fg_flat, fg_flat)
        
        # 重塑回原始形狀
        attended_fg = attended_fg.transpose(1, 2).reshape(b, c, h, w)
        attended_bg = attended_bg.transpose(1, 2).reshape(b, c, h, w)
        
        # 融合特徵
        fused = torch.cat([attended_bg, attended_fg], dim=1)
        fused = self.fusion_layer(fused)
        
        return fused, attended_bg, attended_fg
    
    def align_features(self, fused_features, original_shape):
        """對齊特徵到原始尺寸"""
        aligned = self.alignment_layer(fused_features)
        # 上採樣到原始尺寸
        aligned = F.interpolate(aligned, size=original_shape, mode='bilinear', align_corners=False)
        return aligned
    
    def forward(self, latent_blend, latent_fg):
        """
        前向傳播
        Args:
            latent_blend: 混合圖像的潛在表示 (B, 4, H, W)
            latent_fg: 前景（異常）的潛在表示 (B, 4, H, W)
        Returns:
            interaction_loss: 交互損失
            aligned_features: 對齊後的特徵
        """
        # 提取特徵
        bg_features = self.extract_background_features(latent_blend)
        fg_features = self.extract_shape_features(latent_fg)
        
        # 特徵交互
        fused_features, attended_bg, attended_fg = self.feature_interaction(bg_features, fg_features)
        
        # 對齊特徵
        original_shape = latent_blend.shape[-2:]
        aligned_features = self.align_features(fused_features, original_shape)
        
        # 計算交互損失
        # 1. 背景一致性損失
        bg_consistency_loss = F.mse_loss(attended_bg, bg_features.detach())
        
        # 2. 形狀保持損失
        shape_preservation_loss = F.mse_loss(attended_fg, fg_features.detach())
        
        # 3. 互補性損失（鼓勵背景和前景特徵互補）
        complementary_loss = -F.cosine_similarity(
            attended_bg.flatten(1), 
            attended_fg.flatten(1), 
            dim=1
        ).mean()
        
        # 總交互損失
        interaction_loss = (
            bg_consistency_loss + 
            shape_preservation_loss + 
            0.1 * complementary_loss
        )
        
        return interaction_loss, aligned_features


class DualAnomalyLoss(nn.Module):
    """
    DualAnoDiff 的損失函數
    結合擴散損失和特徵交互損失
    """
    
    def __init__(self, lambda_interaction=0.1):
        super().__init__()
        self.lambda_interaction = lambda_interaction
        
    def forward(self, 
                diffusion_loss_blend, 
                diffusion_loss_fg, 
                interaction_loss,
                mask_alignment_loss=None):
        """
        計算總損失
        Args:
            diffusion_loss_blend: 混合圖像的擴散損失
            diffusion_loss_fg: 前景的擴散損失
            interaction_loss: 特徵交互損失
            mask_alignment_loss: 遮罩對齊損失（可選）
        """
        # 基礎擴散損失
        total_loss = (diffusion_loss_blend + diffusion_loss_fg) / 2.0
        
        # 加入交互損失
        total_loss += self.lambda_interaction * interaction_loss
        
        # 如果有遮罩對齊損失
        if mask_alignment_loss is not None:
            total_loss += 0.05 * mask_alignment_loss
            
        return total_loss