"""
DINOv2 Perceptual Loss

This module implements perceptual loss using pretrained DINOv2 features.
DINOv2 provides rich semantic features that can guide image generation
to maintain perceptual quality.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional

# Add dinov2 to path
DINOV2_PATH = Path(__file__).parent.parent.parent.parent / "extra_repository" / "dinov2"
if str(DINOV2_PATH) not in sys.path:
    sys.path.insert(0, str(DINOV2_PATH))


class DINOv2PerceptualLoss(nn.Module):
    """
    Perceptual loss using DINOv2 features.

    Args:
        model_name: DINOv2 model variant ('vits14', 'vitb14', 'vitl14', 'vitg14')
        feature_layers: List of layer indices to extract features from
        layer_weights: Weights for each layer's contribution to the loss
        loss_type: Type of loss ('l1', 'l2', 'smooth_l1')
        normalize_features: Whether to normalize features before computing loss
        resize_input: Whether to resize input to match DINOv2's expected size
    """

    def __init__(
        self,
        model_name: str = 'vitb14',
        feature_layers: Optional[List[int]] = None,
        layer_weights: Optional[List[float]] = None,
        loss_type: str = 'l2',
        normalize_features: bool = True,
        resize_input: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()

        self.model_name = model_name
        self.loss_type = loss_type
        self.normalize_features = normalize_features
        self.resize_input = resize_input
        self.device = device

        # Default feature layers (intermediate blocks)
        if feature_layers is None:
            feature_layers = [3, 6, 9, 11]  # Extract from multiple transformer blocks
        self.feature_layers = feature_layers

        # Default equal weights for all layers
        if layer_weights is None:
            layer_weights = [1.0] * len(feature_layers)
        assert len(layer_weights) == len(feature_layers), "Number of weights must match number of layers"
        self.layer_weights = layer_weights

        # Load DINOv2 model
        self._load_model()

        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def _load_model(self):
        """Load pretrained DINOv2 model"""
        try:
            from dinov2.hub.backbones import (
                dinov2_vits14,
                dinov2_vitb14,
                dinov2_vitl14,
                dinov2_vitg14
            )

            model_dict = {
                'vits14': dinov2_vits14,
                'vitb14': dinov2_vitb14,
                'vitl14': dinov2_vitl14,
                'vitg14': dinov2_vitg14
            }

            if self.model_name not in model_dict:
                raise ValueError(f"Unknown model: {self.model_name}. Choose from {list(model_dict.keys())}")

            print(f"Loading DINOv2 model: {self.model_name}")
            self.model = model_dict[self.model_name](pretrained=True)
            self.model.to(self.device)
            print(f"DINOv2 model loaded successfully on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load DINOv2 model: {e}")

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for DINOv2.
        DINOv2 expects input in range [0, 1] with ImageNet normalization.

        Args:
            x: Input tensor, assumed to be in range [-1, 1] from VAE decoder

        Returns:
            Normalized input tensor
        """
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0

        # Resize if needed (DINOv2 works with various sizes, but 518x518 is optimal)
        if self.resize_input:
            # DINOv2 expects sizes divisible by patch_size (14)
            target_size = 518  # 37 * 14
            if x.shape[-2:] != (target_size, target_size):
                x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x

    def extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-layer features from DINOv2.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            List of feature tensors from specified layers
        """
        features = []

        # Prepare input
        x = self._prepare_input(x)

        # Extract patch embeddings
        x = self.model.prepare_tokens_with_masks(x)

        # Forward through transformer blocks and collect features
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.feature_layers:
                # Remove CLS token and reshape to spatial format
                # x shape: [B, N+1, D] where N is number of patches
                feat = x[:, 1:, :]  # Remove CLS token

                # Reshape to spatial grid
                B, N, D = feat.shape
                H = W = int(N ** 0.5)
                feat = feat.reshape(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

                # Normalize features if requested
                if self.normalize_features:
                    feat = F.normalize(feat, p=2, dim=1)

                features.append(feat)

        return features

    def compute_loss(
        self,
        pred_features: List[torch.Tensor],
        target_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute perceptual loss between predicted and target features.

        Args:
            pred_features: List of predicted feature tensors
            target_features: List of target feature tensors

        Returns:
            Weighted perceptual loss
        """
        loss = 0.0

        for pred_feat, target_feat, weight in zip(pred_features, target_features, self.layer_weights):
            # Ensure features have the same spatial size
            if pred_feat.shape != target_feat.shape:
                target_feat = F.interpolate(
                    target_feat,
                    size=pred_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Compute loss based on type
            if self.loss_type == 'l1':
                layer_loss = F.l1_loss(pred_feat, target_feat)
            elif self.loss_type == 'l2':
                layer_loss = F.mse_loss(pred_feat, target_feat)
            elif self.loss_type == 'smooth_l1':
                layer_loss = F.smooth_l1_loss(pred_feat, target_feat)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")

            loss += weight * layer_loss

        return loss

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DINOv2 perceptual loss.

        Args:
            pred: Predicted images [B, C, H, W], range [-1, 1]
            target: Target images [B, C, H, W], range [-1, 1]

        Returns:
            Perceptual loss scalar
        """
        # Extract features
        with torch.no_grad():
            target_features = self.extract_features(target)

        pred_features = self.extract_features(pred)

        # Compute loss
        loss = self.compute_loss(pred_features, target_features)

        return loss

    def __repr__(self):
        return (
            f"DINOv2PerceptualLoss(model={self.model_name}, "
            f"layers={self.feature_layers}, "
            f"weights={self.layer_weights}, "
            f"loss_type={self.loss_type})"
        )


def test_dinov2_loss():
    """Test function for DINOv2 perceptual loss"""
    print("Testing DINOv2 Perceptual Loss...")

    # Create dummy data
    batch_size = 2
    pred = torch.randn(batch_size, 3, 512, 512).cuda() * 2 - 1  # Range [-1, 1]
    target = torch.randn(batch_size, 3, 512, 512).cuda() * 2 - 1

    # Initialize loss
    loss_fn = DINOv2PerceptualLoss(
        model_name='vitb14',
        feature_layers=[3, 6, 9, 11],
        layer_weights=[1.0, 1.0, 1.0, 1.0],
        loss_type='l2'
    )

    # Compute loss
    loss = loss_fn(pred, target)
    print(f"Perceptual loss: {loss.item():.4f}")

    # Test gradient flow
    loss.backward()
    print("Gradient test passed!")

    print("Test completed successfully!")


if __name__ == "__main__":
    test_dinov2_loss()
