# Fast-VQA / FasterVQA Research Documentation

## Overview
- **Purpose**: Transformer-based VQA with fragment sampling for real-time performance
- **Key features**:
  - 14× real-time on Apple M1 CPU (FasterVQA)
  - Fragment-based sampling strategy
  - Two variants: Fast-VQA (8 fragments) and FasterVQA (4 fragments)
- **Use cases**: Real-time quality monitoring, streaming applications, large-scale processing

## Technical Details
- **Algorithm**: Vision Transformer with fragment attention
- **Architecture**:
  - Fast-VQA: Swin-Base backbone, 8 fragments
  - FasterVQA: Swin-Tiny backbone, 4 fragments
  - Temporal pooling across fragments
  - Single quality score output
- **Input requirements**:
  - Any resolution (resized to 224×224)
  - 8 frames per fragment
  - Handles variable length videos
- **Output format**:
  - Single quality score (0-1)
  - Inference time metrics

## Implementation Resources
- **Official repository**: https://github.com/VQAssessment/FAST-VQA-and-FasterVQA
- **Papers**:
  - "FAST-VQA: Efficient End-to-end Video Quality Assessment with Fragment Sampling" (ECCV 2022)
  - "FasterVQA: Achieving 14× Speedup" (TPAMI 2023)
- **Documentation**: Well-documented codebase
- **Model weights**:
  - FAST-VQA-B.pth: Base model
  - FasterVQA.pth: Optimized variant

## Implementation Notes
- **Dependencies**:
  - PyTorch, torchvision
  - timm, einops, decord
  - pytorchvideo (optional)
- **Known issues**:
  - Fragment boundary artifacts possible
  - Less accurate on very short videos
- **Performance optimization**:
  - FasterVQA for real-time needs
  - Reduce fragments for speed
  - Cache preprocessed fragments
- **Error handling**:
  - Handle videos shorter than fragment size
  - Validate fragment sampling indices

## Testing
- **Test cases**:
  - Speed benchmark: < 100ms per video ideal
  - Quality correlation with DOVER
  - Fragment consistency tests
- **Expected outputs**:
  - Scores typically 0.3-0.8 range
  - FasterVQA within 5% of Fast-VQA
  - Linear time with video length
- **Validation methods**:
  - Compare with other metrics
  - Measure actual FPS achieved

## Code Snippets
```python
# Fragment sampling
num_fragments = 4 if model_name == "faster-vqa" else 8
fragment_size = len(vreader) // num_fragments

# Model selection
if model_name == "faster-vqa":
    model = FasterVQA()
    fragments = 4
else:
    model = FastVQA()
    fragments = 8

# Preprocessing
frames = frames.float() / 255.0
resize = Resize((224, 224))
```

## References
- ECCV 2022 and TPAMI 2023 papers
- Benchmarked on LIVE-VQC, KoNViD-1k
- Apple M1 optimization techniques