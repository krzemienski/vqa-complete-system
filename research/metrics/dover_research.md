# DOVER Research Documentation

## Overview
- **Purpose**: Disentangled Objective Video Quality Assessment - separates technical quality from aesthetic quality
- **Key features**: 
  - Dual assessment (technical + aesthetic)
  - Mobile variant for CPU efficiency
  - State-of-the-art accuracy on UGC datasets
- **Use cases**: User-generated content, social media videos, general quality assessment

## Technical Details
- **Algorithm**: Swin Transformer backbone with disentangled heads
- **Architecture**:
  - Backbone: Swin-B (full) or Swin-T (mobile)
  - Separate heads for technical and aesthetic quality
  - Temporal aggregation across fragments
- **Input requirements**: 
  - Any resolution video
  - Samples 32 frames across 8 fragments
  - No specific frame rate requirements
- **Output format**:
  - Overall score (0-1)
  - Technical score (0-1) 
  - Aesthetic score (0-1)

## Implementation Resources
- **Official repository**: https://github.com/VQAssessment/DOVER
- **Papers**: 
  - "Disentangling Aesthetic and Technical Effects for Video Quality Assessment of User Generated Content" (ICCV 2023)
- **Documentation**: Comprehensive README with examples
- **Model weights**:
  - DOVER.pth: Full model (~400MB)
  - DOVER-Mobile.pth: Mobile variant (~100MB)

## Implementation Notes
- **Dependencies**:
  - PyTorch 2.1.0+
  - timm, einops, decord
  - CUDA 12 for GPU variant
- **Known issues**:
  - Memory intensive for 4K videos
  - Decord video reader can fail on corrupted files
- **Performance optimization**:
  - Use mobile variant for CPU-only systems
  - Batch multiple fragments for GPU efficiency
  - Pre-resize videos if memory constrained
- **Error handling**:
  - Catch decord failures
  - Handle videos with < 32 frames
  - Validate score ranges

## Testing
- **Test cases**:
  - High quality (BBB 1080p): expect 0.7-0.9
  - Compressed/degraded: expect 0.2-0.4
  - Technical > aesthetic for compression artifacts
- **Expected outputs**:
  - All scores in [0, 1] range
  - Technical and aesthetic can differ significantly
  - Mobile slightly less accurate than full
- **Validation methods**:
  - Compare rankings with human scores
  - Verify degraded < original quality

## Code Snippets
```python
# Key preprocessing
sampler = UnifiedFrameSampler(
    fsize_t=32,      # frames per fragment
    fragments_t=8,   # number of fragments
    num_clips=1      # single clip
)

# Model variants
if model_type == "mobile":
    backbone = "swin_tiny_grpb"
    weights = "DOVER-Mobile.pth"
else:
    backbone = "swin_base_grpb"
    weights = "DOVER.pth"

# Output structure
{
    "overall": float,
    "technical": float,
    "aesthetic": float
}
```

## References
- VQAssessment GitHub organization
- ICCV 2023 paper
- Pre-trained on multiple UGC datasets