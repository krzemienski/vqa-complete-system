# CAMBI Research Documentation

## Overview
- **Purpose**: Netflix's banding artifact detector - specifically targets color banding in compressed video
- **Key features**:
  - Part of VMAF toolkit
  - Focuses on smooth gradient areas
  - Fast CPU-based detection
- **Use cases**: Encoding quality control, banding-prone content (animations, dark scenes), streaming optimization

## Technical Details
- **Algorithm**: Contrast-aware multiscale banding index
- **Architecture**:
  - Analyzes local contrast in smooth regions
  - Multi-scale spatial analysis
  - Temporal consistency checks
  - Outputs banding visibility score
- **Input requirements**:
  - YUV or RGB video
  - Any resolution
  - Works frame-by-frame
- **Output format**:
  - CAMBI score (0-100, higher = more banding)
  - Can normalize to 0-1 range

## Implementation Resources
- **Official repository**: https://github.com/Netflix/vmaf (included in VMAF)
- **Papers**:
  - "CAMBI: Contrast-aware Multiscale Banding Index" (Netflix Tech Blog)
- **Documentation**: VMAF documentation includes CAMBI
- **Model weights**: No weights needed (algorithmic)

## Implementation Notes
- **Dependencies**:
  - VMAF library or vmaf Python package
  - FFmpeg for video handling
- **Known issues**:
  - False positives on film grain
  - Requires careful threshold tuning
- **Performance optimization**:
  - Frame-level parallelization
  - ROI-based processing
- **Error handling**:
  - Validate input color space
  - Handle HDR content properly

## Testing
- **Test cases**:
  - Synthetic gradients with known banding
  - Dark scenes with subtle gradients
  - High-quality content (low scores expected)
- **Expected outputs**:
  - Banding-heavy: 60-100
  - Clean gradients: 0-30
  - Typical content: 20-50
- **Validation methods**:
  - Visual inspection of detected regions
  - Correlation with bitrate

## Code Snippets
```python
# Using VMAF with CAMBI
cmd = [
    "vmaf",
    "--reference", video_path,
    "--distorted", video_path,  # self-reference for NR
    "--cambi",
    "--output", output_json,
    "--json"
]

# Score extraction
cambi_score = results["aggregate"]["CAMBI"]
normalized = cambi_score / 100.0  # to 0-1
```

## References
- Netflix Technology Blog
- VMAF GitHub repository
- Picture Quality Symposium presentations