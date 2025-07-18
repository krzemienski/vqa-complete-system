# Video Quality Assessment Fundamentals

## No-Reference vs Full-Reference VQA

### No-Reference (NR) / Blind VQA
- Assesses video quality without access to original/reference video
- Uses learned features and statistical models to predict quality
- Essential for real-world applications where reference is unavailable
- All 13 metrics in this system are NR-VQA methods

### Full-Reference (FR) VQA
- Requires pristine reference video for comparison
- Examples: PSNR, SSIM, VMAF
- More accurate but less practical for many use cases

## Common Quality Artifacts

### 1. Compression Artifacts
- **Blockiness**: Visible block boundaries from DCT-based compression
- **Blur**: Loss of high-frequency details
- **Ringing**: Oscillations near sharp edges
- **Color bleeding**: Color information spreading beyond boundaries

### 2. Noise Types
- **Gaussian noise**: Random pixel variations
- **Salt-and-pepper noise**: Random black/white pixels
- **Film grain**: Natural texture in analog sources
- **Sensor noise**: From camera sensors in low light

### 3. Banding
- **Color banding**: Visible steps in gradients
- **Detected by**: CAMBI metric specifically
- **Common in**: Low bitrate encoding, 8-bit content

### 4. Motion Artifacts
- **Judder**: Uneven motion from frame rate issues
- **Motion blur**: From camera/object movement
- **Shake/Jitter**: From handheld recording
- **Detected by**: StableVQA, TLVQM

## Performance vs Accuracy Tradeoffs

### Fast Metrics (Real-time capable)
- **DOVER-Mobile**: ~1.4s per 1080p clip on CPU
- **FasterVQA**: 14× real-time on Apple M1
- **MDTVSFA**: Good balance of speed/accuracy
- **Trade-off**: Slightly lower accuracy for speed

### Accurate but Slower Metrics
- **DOVER-Full**: GPU required, higher accuracy
- **Video-BLIINDS**: Very slow (5+ min/video) but thorough
- **VIDEVAL**: 60 features, comprehensive but slower
- **Trade-off**: Better accuracy at cost of processing time

### Specialized Metrics
- **CAMBI**: Fast, specific to banding detection
- **StableVQA**: Focuses on stabilization quality
- **TLVQM**: Targets mobile/consumer video artifacts

## GPU/CPU Considerations

### GPU-Accelerated Metrics
- **DOVER-Full**: Requires CUDA, 5-10× speedup
- **COVER**: Ensemble model benefits from GPU
- **Fast-VQA**: Optional GPU support

### CPU-Only Metrics
- **DOVER-Mobile**: Optimized for CPU
- **FasterVQA**: Efficient on CPU
- **CAMBI**: CPU-based banding detection
- **MATLAB metrics**: Typically CPU-bound

### Memory Requirements
- **4K videos**: May require 16GB+ RAM
- **Batch processing**: Memory scales with parallel jobs
- **Docker overhead**: Add 2-4GB per container

## Metric Selection Guidelines

### For General Quality Assessment
- Use DOVER (mobile for speed, full for accuracy)
- Add Fast-VQA/FasterVQA for transformer-based assessment
- Include MDTVSFA for cross-dataset robustness

### For Specific Artifacts
- **Banding**: CAMBI is specialized for this
- **Mobile/shake**: StableVQA and TLVQM
- **Compression artifacts**: Most metrics detect these

### For Research/Comprehensive Analysis
- Run all available metrics
- Use orchestrator for parallel execution
- Generate comparative reports

## Implementation Best Practices

### 1. Standardized Output
- All metrics output JSON with consistent schema
- Include metric name, scores, timing, status
- Handle errors gracefully with error field

### 2. Docker Containerization
- Self-contained with all dependencies
- Pre-download model weights during build
- Use multi-stage builds to reduce size

### 3. Testing Strategy
- Test with known good/bad videos
- Verify score ranges and relationships
- Include edge cases (very short, corrupted)

### 4. Performance Optimization
- Cache preprocessed frames when possible
- Use appropriate batch sizes
- Parallelize across videos, not metrics