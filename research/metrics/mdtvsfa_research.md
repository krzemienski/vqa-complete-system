# MDTVSFA Research Documentation

## Overview
- **Purpose**: Mixed-Dataset Training for Video quality assessment with Strong cross-dataset performance
- **Key features**:
  - Trained on multiple VQA datasets simultaneously
  - Robust generalization across content types
  - 3D-CNN architecture with temporal pooling
- **Use cases**: Cross-platform quality assessment, diverse content types, research benchmarking

## Technical Details
- **Algorithm**: 3D Convolutional Neural Network with mixed training
- **Architecture**:
  - 5 3D-Conv blocks with batch normalization
  - Adaptive 3D pooling
  - Fully connected regression layers
  - Temporal clip aggregation
- **Input requirements**:
  - Extracts 10 clips of 16 frames each
  - Resizes to 112×112
  - Handles variable length videos
- **Output format**:
  - Overall quality score (0-1)
  - Individual clip scores
  - Standard deviation across clips

## Implementation Resources
- **Official repository**: https://github.com/lidq92/MDTVSFA
- **Papers**:
  - "Unified Quality Assessment of In-the-Wild Videos with Mixed Datasets Training" (IJCV 2021)
- **Documentation**: GitHub README with training details
- **Model weights**:
  - MDTVSFA.pt: Pre-trained model

## Implementation Notes
- **Dependencies**:
  - PyTorch, torchvision
  - OpenCV for frame extraction
  - scikit-learn for preprocessing
- **Known issues**:
  - High memory usage for clip storage
  - Slower than transformer methods
- **Performance optimization**:
  - Reduce clip count for speed
  - Use weighted aggregation
  - Parallel clip processing
- **Error handling**:
  - Handle videos < 16 frames
  - Clip boundary validation

## Testing
- **Test cases**:
  - Cross-dataset validation
  - Clip consistency checks
  - Aggregation method comparison
- **Expected outputs**:
  - Normalized scores via sigmoid
  - Clip std indicates temporal consistency
  - Works on diverse content
- **Validation methods**:
  - Test on unseen datasets
  - Compare aggregation strategies

## Code Snippets
```python
# Architecture
self.features = nn.Sequential(
    nn.Conv3d(3, 16, kernel_size=(5,5,5), padding=2),
    nn.BatchNorm3d(16),
    nn.ReLU(inplace=True),
    nn.MaxPool3d(kernel_size=(2,2,2)),
    # ... more layers
    nn.AdaptiveAvgPool3d((1,1,1))
)

# Clip extraction
num_clips = 10
clip_len = 16
clips = extract_clips(video, num_clips, clip_len)

# Score aggregation
if aggregate == "weighted":
    weights = gaussian_weights(num_clips)
    final_score = np.sum(clip_scores * weights)
```

## References
- IJCV 2021 paper
- Trained on LIVE-VQC + KoNViD-1k + YouTube-UGC
- Cross-dataset evaluation protocol