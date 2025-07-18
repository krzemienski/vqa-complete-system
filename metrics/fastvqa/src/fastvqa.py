"""
Fast-VQA and FasterVQA model implementations.
Fragment-based video quality assessment with Vision Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import timm


class FragmentSampler(nn.Module):
    """Sample and process video fragments."""
    
    def __init__(self, num_fragments=8, frames_per_fragment=8):
        super().__init__()
        self.num_fragments = num_fragments
        self.frames_per_fragment = frames_per_fragment
    
    def forward(self, x):
        """
        Args:
            x: (B, F, T, C, H, W) - Batch, Fragments, Time, Channels, Height, Width
        Returns:
            x: (B*F, T, C, H, W) - Flattened for processing
        """
        B, F, T, C, H, W = x.shape
        x = rearrange(x, 'b f t c h w -> (b f) t c h w')
        return x, B, F


class VisionTransformerBackbone(nn.Module):
    """Vision Transformer backbone with temporal modifications."""
    
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.vit.num_features
        
    def forward(self, x):
        """Process frames through ViT."""
        # x: (BF, T, C, H, W)
        BF, T, C, H, W = x.shape
        
        # Process each frame
        x = rearrange(x, 'bf t c h w -> (bf t) c h w')
        features = self.vit(x)  # (BF*T, D)
        
        # Reshape back
        features = rearrange(features, '(bf t) d -> bf t d', bf=BF, t=T)
        
        return features


class SwinTransformerBackbone(nn.Module):
    """Swin Transformer backbone for FasterVQA."""
    
    def __init__(self, model_name='swin_tiny_patch4_window7_224', pretrained=True):
        super().__init__()
        self.swin = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.swin.num_features
        
    def forward(self, x):
        """Process frames through Swin."""
        BF, T, C, H, W = x.shape
        
        # Process each frame
        x = rearrange(x, 'bf t c h w -> (bf t) c h w')
        features = self.swin(x)  # (BF*T, D)
        
        # Reshape back
        features = rearrange(features, '(bf t) d -> bf t d', bf=BF, t=T)
        
        return features


class TemporalAttentionPooling(nn.Module):
    """Temporal attention pooling for aggregating frame features."""
    
    def __init__(self, feat_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (BF, T, D) - Features for each frame
        Returns:
            pooled: (BF, D) - Temporally pooled features
        """
        # Compute attention weights
        weights = self.attention(x)  # (BF, T, 1)
        weights = F.softmax(weights, dim=1)
        
        # Apply attention
        pooled = torch.sum(x * weights, dim=1)  # (BF, D)
        
        return pooled


class QualityRegressor(nn.Module):
    """Final quality score regression head."""
    
    def __init__(self, feat_dim, hidden_dim=512):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        """Predict quality score."""
        return torch.sigmoid(self.regressor(x)).squeeze(-1)


class FastVQA(nn.Module):
    """Fast-VQA: Fragment-based VQA with ViT backbone."""
    
    def __init__(self, num_fragments=8, frames_per_fragment=8):
        super().__init__()
        self.fragment_sampler = FragmentSampler(num_fragments, frames_per_fragment)
        self.backbone = VisionTransformerBackbone('vit_base_patch16_224')
        self.temporal_pooling = TemporalAttentionPooling(self.backbone.feat_dim)
        self.quality_head = QualityRegressor(self.backbone.feat_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, F, T, C, H, W) - Video fragments
        Returns:
            score: (B,) - Quality scores
        """
        # Sample fragments
        x, B, F = self.fragment_sampler(x)
        
        # Extract features
        features = self.backbone(x)  # (B*F, T, D)
        
        # Temporal pooling
        fragment_features = self.temporal_pooling(features)  # (B*F, D)
        
        # Reshape to separate batch and fragments
        fragment_features = rearrange(fragment_features, '(b f) d -> b f d', b=B, f=F)
        
        # Average across fragments
        video_features = fragment_features.mean(dim=1)  # (B, D)
        
        # Predict quality
        score = self.quality_head(video_features)  # (B,)
        
        return score


class FasterVQA(nn.Module):
    """FasterVQA: Optimized version with Swin-Tiny and fewer fragments."""
    
    def __init__(self, num_fragments=4, frames_per_fragment=8):
        super().__init__()
        self.fragment_sampler = FragmentSampler(num_fragments, frames_per_fragment)
        self.backbone = SwinTransformerBackbone('swin_tiny_patch4_window7_224')
        self.temporal_pooling = TemporalAttentionPooling(self.backbone.feat_dim)
        self.quality_head = QualityRegressor(self.backbone.feat_dim, hidden_dim=256)
        
    def forward(self, x):
        """Same as FastVQA but with optimized architecture."""
        # Sample fragments
        x, B, F = self.fragment_sampler(x)
        
        # Extract features
        features = self.backbone(x)
        
        # Temporal pooling
        fragment_features = self.temporal_pooling(features)
        
        # Reshape and average
        fragment_features = rearrange(fragment_features, '(b f) d -> b f d', b=B, f=F)
        video_features = fragment_features.mean(dim=1)
        
        # Predict quality
        score = self.quality_head(video_features)
        
        return score