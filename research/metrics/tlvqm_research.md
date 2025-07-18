# CNN-TLVQM Research Documentation

## Overview
- **Purpose**: Two-Level Video Quality Model focusing on compression and capture artifacts
- **Key features**:
  - CNN variant: Deep learning enhancement
  - Original TLVQM: Feature-based approach
  - Specialized for mobile/consumer video
- **Use cases**: Mobile video assessment, compression artifact detection, capture quality evaluation

## Technical Details
- **Algorithm**: Two-level quality assessment (spatial + temporal)
- **Architecture**:
  - Level 1: Frame-level quality (blur, noise, blockiness)
  - Level 2: Temporal quality (shake, flicker, motion)
  - CNN variant: ResNet backbone with quality heads
  - Feature aggregation across levels
- **Input requirements**:
  - RGB video, any resolution
  - Works best with 5+ seconds
  - Handles variable frame rates
- **Output format**:
  - Overall quality score
  - Spatial quality component
  - Temporal quality component
  - Individual artifact scores

## Implementation Resources
- **Official repository**: https://github.com/jarikorhonen/CNN-TLVQM
- **Papers**:
  - "CNN-TLVQM - Two-Level Video Quality Model using Space-Time Pooling" (2019)
  - Original TLVQM paper (2015)
- **Documentation**: Implementation guides for both variants
- **Model weights**:
  - CNN-TLVQM.pth: Neural network weights
  - TLVQM parameters in code

## Implementation Notes
- **Dependencies**:
  - PyTorch (CNN variant)
  - OpenCV, NumPy
  - scikit-image for features
  - Optional: CUDA for CNN
- **Known issues**:
  - Sensitive to extreme shake
  - May overpenalize intentional motion
  - CNN variant needs GPU for speed
- **Performance optimization**:
  - Frame sampling for efficiency
  - Batch processing in CNN
  - ROI-based quality assessment
- **Error handling**:
  - Motion estimation failures
  - Extreme parameter validation
  - Fallback to feature-based

## Testing
- **Test cases**:
  - Handheld mobile video
  - Various compression levels
  - Synthetic shake/blur
  - Professional vs amateur content
- **Expected outputs**:
  - Mobile video: typically 40-70
  - Professional: 70-90
  - Heavy artifacts: 20-40
- **Validation methods**:
  - Artifact injection tests
  - Comparison with TLVQM original
  - Mobile dataset validation

## Code Snippets
```python
# CNN-TLVQM architecture
class CNNTLVQM(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial stream
        self.spatial_backbone = resnet50(pretrained=True)
        self.spatial_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # blur, noise, block
        )
        
        # Temporal stream
        self.temporal_conv = nn.Conv3d(3, 64, (5,3,3))
        self.temporal_head = nn.Sequential(
            nn.Linear(64*4*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # shake, flicker
        )
        
        # Fusion
        self.fusion = nn.Linear(5, 1)

# Feature-based TLVQM
def compute_tlvqm_features(video):
    # Spatial features
    blur_score = compute_blur(video)
    noise_score = compute_noise(video)
    block_score = compute_blockiness(video)
    
    # Temporal features
    shake_score = compute_shake(video)
    flicker_score = compute_flicker(video)
    
    # Two-level aggregation
    spatial_quality = aggregate_spatial(blur, noise, block)
    temporal_quality = aggregate_temporal(shake, flicker)
    
    overall = 0.6 * spatial_quality + 0.4 * temporal_quality
    return overall
```

## References
- ICIP 2019 (CNN variant)
- Original TLVQM: IEEE CSVT 2015
- Mobile video quality datasets
- Consumer video artifact studies