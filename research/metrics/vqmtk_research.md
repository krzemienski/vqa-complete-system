# VQMTK Research Documentation

## Overview
- **Purpose**: Video Quality Metrics Toolkit - collection of 14 classical VQA metrics in one package
- **Key features**:
  - Includes BRISQUE, NIQE, VIIDEO, etc.
  - Consistent interface across metrics
  - Both NR and FR metrics available
- **Use cases**: Comprehensive quality analysis, metric comparison, research baseline

## Technical Details
- **Algorithm**: Multiple algorithms in one toolkit
- **Available metrics**:
  - No-Reference: BRISQUE, NIQE, VIIDEO, CORNIA
  - Full-Reference: SSIM, MS-SSIM, PSNR, VIF
  - Video-specific: V-BLIINDS, TLVQM variants
  - 14 total metrics with unified API
- **Input requirements**:
  - Video files (MP4, AVI, MOV)
  - Some metrics need specific preprocessing
  - FR metrics need reference video
- **Output format**:
  - JSON with all requested metrics
  - Individual metric scores
  - Computation time per metric

## Implementation Resources
- **Official repository**: https://github.com/VQEG/VQMTK
- **Papers**:
  - Individual papers for each metric
  - VQEG standardization documents
- **Documentation**: Unified API documentation
- **Model weights**:
  - Pre-trained models for each metric
  - Calibration parameters included

## Implementation Notes
- **Dependencies**:
  - Python 3.8+
  - OpenCV, NumPy, SciPy
  - scikit-image, scikit-video
  - Optional: MATLAB engine for some metrics
- **Known issues**:
  - Some metrics incompatible with others
  - Memory usage with multiple metrics
  - Speed varies greatly between metrics
- **Performance optimization**:
  - Selective metric execution
  - Shared preprocessing pipeline
  - Result caching system
- **Error handling**:
  - Metric-specific error catching
  - Fallback to available metrics
  - Detailed error reporting

## Testing
- **Test cases**:
  - Run all 14 metrics on test video
  - Compare with individual implementations
  - Memory usage monitoring
  - Speed benchmarking
- **Expected outputs**:
  - Each metric in expected range
  - Consistent JSON structure
  - No metric conflicts
- **Validation methods**:
  - Cross-validation with papers
  - Regression testing
  - Performance profiling

## Code Snippets
```python
# VQMTK unified interface
from vqmtk import VQMTK

# Initialize toolkit
vqm = VQMTK()

# Configure metrics to run
config = {
    "metrics": ["brisque", "niqe", "viideo", "tlvqm"],
    "preprocessing": {
        "resize": None,
        "color_space": "rgb"
    },
    "output": {
        "format": "json",
        "include_timing": True
    }
}

# Run assessment
results = vqm.assess(video_path, config)

# Results structure
{
    "video": "test.mp4",
    "metrics": {
        "brisque": {
            "score": 45.2,
            "time": 1.23
        },
        "niqe": {
            "score": 4.56,
            "time": 2.34
        },
        "viideo": {
            "score": 0.67,
            "time": 5.67
        },
        "tlvqm": {
            "score": 72.3,
            "spatial": 68.9,
            "temporal": 75.7,
            "time": 3.45
        }
    },
    "total_time": 12.69
}

# Selective execution
nr_metrics = vqm.get_nr_metrics()
fast_metrics = vqm.get_fast_metrics()

# Batch processing
videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
batch_results = vqm.assess_batch(videos, config)
```

## References
- VQEG (Video Quality Experts Group)
- Individual metric papers
- Standardization efforts
- Benchmarking studies