# Objective-Metrics CLI Research Documentation

## Overview
- **Purpose**: Unified command-line interface providing access to 50+ video quality metrics
- **Key features**:
  - Extensive metric collection (NR and FR)
  - Consistent CLI interface
  - Batch processing capabilities
  - Multiple output formats
- **Use cases**: Comprehensive quality analysis, metric comparison studies, production pipelines

## Technical Details
- **Algorithm**: Wrapper around multiple VQA implementations
- **Available metrics**:
  - No-Reference: BRISQUE, NIQE, BLIINDS, CORNIA, DIIVINE, HIGRADE, etc.
  - Full-Reference: PSNR, SSIM, MS-SSIM, VIF, VMAF, etc.
  - Video-specific: VIIDEO, V-BLIINDS, temporal metrics
  - Perceptual: LPIPS, DISTS, PieAPP
  - 50+ total metrics accessible
- **Input requirements**:
  - Standard video formats
  - Automatic format conversion
  - Handles various codecs
- **Output format**:
  - JSON, CSV, XML options
  - Metric aggregation
  - Statistical summaries

## Implementation Resources
- **Official repository**: https://github.com/objective-metrics/objective-metrics-cli
- **Papers**:
  - References for each included metric
- **Documentation**: Comprehensive CLI guide
- **Model weights**:
  - Auto-downloaded on first use
  - Cached locally

## Implementation Notes
- **Dependencies**:
  - Python 3.8+
  - FFmpeg (required)
  - Metric-specific dependencies
  - Optional: GPU libraries
- **Known issues**:
  - Large download on first run
  - Some metrics conflict
  - Version compatibility issues
- **Performance optimization**:
  - Metric selection flags
  - Parallel video processing
  - Result caching system
  - GPU acceleration where available
- **Error handling**:
  - Graceful metric failures
  - Dependency checking
  - Format validation

## Testing
- **Test cases**:
  - Run subset of fast metrics
  - Format conversion testing
  - Batch processing validation
  - Output format verification
- **Expected outputs**:
  - Consistent with individual metrics
  - Proper error reporting
  - Valid output formats
- **Validation methods**:
  - Compare with standalone metrics
  - Regression testing
  - Performance benchmarking

## Code Snippets
```python
# CLI usage examples
"""
# Run specific metrics
objective-metrics -i video.mp4 -m brisque niqe viideo -o results.json

# Run all no-reference metrics
objective-metrics -i video.mp4 --nr-only -o nr_results.csv

# Batch processing
objective-metrics -i videos/*.mp4 -m brisque -o batch_results.json

# With reference video
objective-metrics -i distorted.mp4 -r reference.mp4 -m ssim vmaf -o fr_results.json

# Custom configuration
objective-metrics -i video.mp4 -c custom_config.yaml -o results.xml
"""

# Python API wrapper
import objective_metrics as om

# Initialize
analyzer = om.VideoAnalyzer()

# Configure metrics
config = {
    'metrics': ['brisque', 'niqe', 'viideo'],
    'output_format': 'json',
    'preprocessing': {
        'resize': None,
        'fps': None
    }
}

# Single video analysis
results = analyzer.analyze('video.mp4', config)

# Batch processing
videos = ['video1.mp4', 'video2.mp4', 'video3.mp4']
batch_results = analyzer.analyze_batch(videos, config)

# Custom metric selection
nr_metrics = analyzer.get_available_metrics(type='no-reference')
fast_metrics = analyzer.get_fast_metrics(max_time=5.0)

# Results structure
{
    "file": "video.mp4",
    "duration": 120.5,
    "resolution": "1920x1080",
    "metrics": {
        "brisque": {
            "score": 45.67,
            "computation_time": 2.34,
            "status": "success"
        },
        "niqe": {
            "score": 4.89,
            "computation_time": 3.45,
            "status": "success"
        },
        "viideo": {
            "score": 0.234,
            "computation_time": 8.90,
            "status": "success"
        }
    },
    "total_time": 14.69,
    "timestamp": "2024-01-15T10:30:00Z"
}

# Configuration file format
"""
metrics:
  no_reference:
    - brisque
    - niqe
    - viideo
  full_reference:
    - ssim
    - vmaf
    
preprocessing:
  resize: [1920, 1080]
  color_space: yuv420p
  
output:
  format: json
  include_stats: true
  save_frames: false
  
performance:
  parallel_videos: 4
  gpu_acceleration: true
  cache_results: true
"""
```

## References
- Individual metric papers
- FFmpeg documentation
- Video quality assessment surveys
- CLI design best practices