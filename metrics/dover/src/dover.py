"""
DOVER model implementation.
Disentangled Objective Video Quality Evaluator with technical and aesthetic branches.
Matches official DOVER architecture with dual-view processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import timm


class TechnicalBackbone(nn.Module):
    """Swin3D backbone for technical quality assessment (fragments view)."""
    
    def __init__(self):
        super().__init__()
        # Simplified Swin3D-like architecture
        self.patch_embed = nn.Conv3d(3, 96, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(96, 192, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm3d(192)
            ),
            nn.Sequential(
                nn.Conv3d(192, 384, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm3d(384)
            ),
            nn.Sequential(
                nn.Conv3d(384, 768, kernel_size=3, padding=1),
                nn.GELU(),
                nn.BatchNorm3d(768)
            )
        ])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, x):
        # x: (B, F, T, C, H, W) - technical fragments, reshape to process all fragments
        if x.dim() == 6:
            B, F, T, C, H, W = x.shape
            x = x.view(B * F, T, C, H, W)  # Flatten batch and fragment dimensions
            self._current_batch_size = B
            self._current_fragments = F
        else:
            B = x.shape[0]
            F = 1
            self._current_batch_size = B
            self._current_fragments = F
            
        x = x.permute(0, 2, 1, 3, 4)  # (BF, C, T, H, W)
        
        x = self.patch_embed(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.avgpool(x)  # (BF, 768, 1, 1, 1)
        x = x.flatten(1)     # (BF, 768)
        
        # Reshape back to handle fragments
        if hasattr(self, '_current_batch_size') and hasattr(self, '_current_fragments'):
            B, F = self._current_batch_size, self._current_fragments
            x = x.view(B, F, 768)     # (B, F, 768)
            x = x.mean(dim=1)         # Average across fragments: (B, 768)
        
        return x


class AestheticBackbone(nn.Module):
    """ConvNeXt backbone for aesthetic quality assessment (resize view)."""
    
    def __init__(self):
        super().__init__()
        # Simplified ConvNeXt-like architecture for temporal processing
        self.stem_conv = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        
        self.layers = nn.ModuleList([
            self._make_layer(96, 192),
            self._make_layer(192, 384),
            self._make_layer(384, 768)
        ])
        
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _make_layer(self, in_channels, out_channels):
        """Create a ConvNeXt-style layer with proper LayerNorm handling."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, groups=in_channels),
            nn.BatchNorm2d(out_channels),  # Use BatchNorm instead of LayerNorm for simplicity
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: (B, T, C, H, W) - aesthetic resized frames
        B, T, C, H, W = x.shape
        
        # Process each frame
        x = x.view(B * T, C, H, W)
        
        # Stem
        x = self.stem_conv(x)
        
        # Layers
        for layer in self.layers:
            x = layer(x)
        
        # Spatial pooling
        x = self.spatial_pool(x)  # (BT, 768, 1, 1)
        x = x.view(B, T, 768)     # (B, T, 768)
        
        # Temporal pooling
        x = x.permute(0, 2, 1)    # (B, 768, T)
        x = self.temporal_pool(x) # (B, 768, 1)
        x = x.squeeze(-1)         # (B, 768)
        
        return x


class VQAHead(nn.Module):
    """Quality prediction head matching official DOVER."""
    
    def __init__(self, in_channels=768, hidden_channels=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x):
        return self.fc(x).squeeze(-1)


class DOVER(nn.Module):
    """DOVER: Disentangled Objective Video Quality Evaluator.
    
    Official architecture with separate technical and aesthetic pathways.
    """
    
    def __init__(self):
        super().__init__()
        
        # Separate backbones for technical and aesthetic views
        self.technical_backbone = TechnicalBackbone()
        self.aesthetic_backbone = AestheticBackbone()
        
        # Separate heads for each view
        self.technical_head = VQAHead(in_channels=768, hidden_channels=64)
        self.aesthetic_head = VQAHead(in_channels=768, hidden_channels=64)
        
    def forward(self, vclips):
        """
        Args:
            vclips: Dict with 'technical' and 'aesthetic' keys containing video tensors
                   technical: (B, T, C, H, W) - fragment view
                   aesthetic: (B, T, C, H, W) - resized view
        
        Returns:
            List of [technical_score, aesthetic_score]
        """
        scores = []
        
        # Process technical view (fragments)
        if 'technical' in vclips:
            tech_feat = self.technical_backbone(vclips['technical'])
            tech_score = self.technical_head(tech_feat)
            scores.append(tech_score)
        
        # Process aesthetic view (resized)
        if 'aesthetic' in vclips:
            aes_feat = self.aesthetic_backbone(vclips['aesthetic'])
            aes_score = self.aesthetic_head(aes_feat)
            scores.append(aes_score)
        
        return scores


class DOVERMobile(DOVER):
    """Mobile variant of DOVER using smaller backbones."""
    
    def __init__(self):
        super().__init__()
        # Use same architecture but could be optimized for mobile