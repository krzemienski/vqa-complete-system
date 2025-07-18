"""
MDTVSFA (Mixed Dataset Training VQA) implementation.
Cross-dataset robust video quality assessment through mixed dataset training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50
import timm


class MotionExtractor(nn.Module):
    """Extract motion features from frame differences."""
    
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1)
        self.conv3 = nn.Conv3d(64, out_channels, kernel_size=(3, 3, 3), padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1, 7, 7))
        
    def forward(self, x):
        """Extract motion features from consecutive frames."""
        # Compute frame differences
        diff = x[:, 1:] - x[:, :-1]  # (B, T-1, C, H, W)
        
        # Add channel dimension for 3D conv
        diff = diff.permute(0, 2, 1, 3, 4)  # (B, C, T-1, H, W)
        
        # Extract features
        x = F.relu(self.conv1(diff))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2))
        
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, kernel_size=(1, 2, 2))
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # (B, 128, 1, 7, 7)
        
        return x.squeeze(2)  # (B, 128, 7, 7)


class SpatialFeatureExtractor(nn.Module):
    """Extract spatial features using pre-trained ResNet."""
    
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet50(pretrained=pretrained)
        # Remove the final FC layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.out_channels = 2048
        
    def forward(self, x):
        """Extract spatial features from frames."""
        B, T, C, H, W = x.shape
        
        # Process all frames at once
        x = x.view(B * T, C, H, W)
        features = self.features(x)  # (B*T, 2048, 7, 7)
        
        # Reshape back
        _, C_out, H_out, W_out = features.shape
        features = features.view(B, T, C_out, H_out, W_out)
        
        return features


class TemporalAggregation(nn.Module):
    """Temporal aggregation with attention mechanism."""
    
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, T, D) temporal features
        Returns:
            aggregated: (B, D) aggregated features
        """
        # Compute attention weights
        weights = self.attention(x)  # (B, T, 1)
        weights = F.softmax(weights, dim=1)
        
        # Apply attention
        aggregated = torch.sum(x * weights, dim=1)  # (B, D)
        
        return aggregated


class DatasetDiscriminator(nn.Module):
    """Discriminator for dataset-specific features."""
    
    def __init__(self, feature_dim, num_datasets):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, num_datasets)
        )
        
    def forward(self, x):
        """Predict dataset origin."""
        return self.discriminator(x)


class MDTVSFA(nn.Module):
    """Mixed Dataset Training VQA model."""
    
    def __init__(self, num_datasets=5, backbone='resnet50'):
        super().__init__()
        
        # Feature extractors
        self.spatial_extractor = SpatialFeatureExtractor(pretrained=True)
        self.motion_extractor = MotionExtractor()
        
        # Feature dimensions
        spatial_dim = self.spatial_extractor.out_channels
        motion_dim = 128
        
        # Feature fusion
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.motion_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal modeling
        self.temporal_spatial = TemporalAggregation(spatial_dim)
        self.temporal_motion = TemporalAggregation(motion_dim)
        
        # Dataset discriminator for adversarial training
        self.dataset_discriminator = DatasetDiscriminator(
            spatial_dim + motion_dim, 
            num_datasets
        )
        
        # Quality regression head
        fused_dim = spatial_dim + motion_dim
        self.quality_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
        # Dataset-specific calibration layers
        self.dataset_calibration = nn.ModuleList([
            nn.Linear(1, 1, bias=True) for _ in range(num_datasets)
        ])
        
    def extract_features(self, x):
        """Extract spatial and motion features."""
        # Spatial features
        spatial_features = self.spatial_extractor(x)  # (B, T, 2048, 7, 7)
        B, T, C_s, H_s, W_s = spatial_features.shape
        
        # Pool spatial features
        spatial_features = spatial_features.view(B * T, C_s, H_s, W_s)
        spatial_features = self.spatial_pool(spatial_features).squeeze(-1).squeeze(-1)
        spatial_features = spatial_features.view(B, T, C_s)
        
        # Motion features (need at least 2 frames)
        if T > 1:
            motion_features = self.motion_extractor(x)  # (B, 128, 7, 7)
            motion_features = self.motion_pool(motion_features).squeeze(-1).squeeze(-1)  # (B, 128)
            motion_features = motion_features.unsqueeze(1).expand(B, T, -1)  # (B, T, 128)
        else:
            # For single frame, use zero motion features
            motion_features = torch.zeros(B, T, 128, device=x.device)
        
        return spatial_features, motion_features
        
    def forward(self, x, dataset_id=None, return_features=False):
        """
        Args:
            x: (B, T, C, H, W) video frames
            dataset_id: Dataset identifier for calibration
            return_features: Whether to return intermediate features
        Returns:
            score: (B,) quality scores
            features: Optional dict of intermediate features
        """
        # Extract features
        spatial_features, motion_features = self.extract_features(x)
        
        # Temporal aggregation
        spatial_aggregated = self.temporal_spatial(spatial_features)  # (B, 2048)
        motion_aggregated = self.temporal_motion(motion_features)  # (B, 128)
        
        # Fuse features
        fused_features = torch.cat([spatial_aggregated, motion_aggregated], dim=1)
        
        # Predict quality
        quality_score = self.quality_head(fused_features)  # (B, 1)
        
        # Apply dataset-specific calibration if available
        if dataset_id is not None and dataset_id < len(self.dataset_calibration):
            quality_score = self.dataset_calibration[dataset_id](quality_score)
        
        # Apply sigmoid to ensure [0, 1] range
        quality_score = torch.sigmoid(quality_score).squeeze(-1)
        
        if return_features:
            # Dataset discrimination (for training)
            dataset_pred = self.dataset_discriminator(fused_features.detach())
            
            features = {
                'spatial': spatial_aggregated,
                'motion': motion_aggregated,
                'fused': fused_features,
                'dataset_pred': dataset_pred
            }
            return quality_score, features
        
        return quality_score


class MDTVSFALite(nn.Module):
    """Lightweight version of MDTVSFA using MobileNet backbone."""
    
    def __init__(self, num_datasets=5):
        super().__init__()
        
        # Use MobileNetV3 for efficiency
        self.backbone = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features
        
        # Simplified motion extractor
        self.motion_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.motion_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Quality head
        fused_dim = backbone_dim + 32
        self.quality_head = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        """Simplified forward pass."""
        B, T, C, H, W = x.shape
        
        # Extract spatial features
        x_flat = x.view(B * T, C, H, W)
        spatial_features = self.backbone(x_flat)  # (B*T, backbone_dim)
        spatial_features = spatial_features.view(B, T, -1)
        
        # Extract motion features (frame differences)
        if T > 1:
            diff = x[:, 1:] - x[:, :-1]  # (B, T-1, C, H, W)
            diff = diff.mean(dim=1)  # Average over time
            motion_features = F.relu(self.motion_conv(diff))
            motion_features = self.motion_pool(motion_features).squeeze(-1).squeeze(-1)
        else:
            motion_features = torch.zeros(B, 32, device=x.device)
        
        # Temporal pooling
        spatial_pooled = self.temporal_pool(spatial_features.transpose(1, 2)).squeeze(-1)
        
        # Combine features
        combined = torch.cat([spatial_pooled, motion_features], dim=1)
        
        # Predict quality
        score = torch.sigmoid(self.quality_head(combined)).squeeze(-1)
        
        return score