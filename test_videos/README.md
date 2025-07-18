# Test Videos for VQA System

This directory contains test videos for evaluating the VQA metrics.

## Directory Structure

```
test_videos/
├── original/          # Original high-quality videos
│   ├── bbb_1080p.mp4     # Big Buck Bunny 1080p
│   ├── bbb_720p.mp4      # Big Buck Bunny 720p
│   ├── bbb_480p.mp4      # Big Buck Bunny 480p
│   ├── tos_1080p.mp4     # Tears of Steel 1080p
│   ├── tos_720p.mp4      # Tears of Steel 720p
│   ├── sintel_1080p.mp4  # Sintel 1080p
│   └── sintel_720p.mp4   # Sintel 720p
│
└── degraded/          # Degraded versions for testing
    ├── compression/      # H.264 compression artifacts
    │   ├── *_crf40.mp4      # CRF 40 (high compression)
    │   ├── *_crf50.mp4      # CRF 50 (very high compression)
    │   ├── *_500kbps.mp4    # 500 kbps bitrate
    │   └── *_200kbps.mp4    # 200 kbps bitrate
    │
    ├── noise/           # Various noise types
    │   ├── *_noise_low.mp4  # Low Gaussian noise
    │   ├── *_noise_high.mp4 # High Gaussian noise
    │   └── *_salt_pepper.mp4 # Salt and pepper noise
    │
    ├── blur/            # Blur effects
    │   ├── *_blur_low.mp4   # Low blur (radius 2)
    │   ├── *_blur_high.mp4  # High blur (radius 5)
    │   └── *_motion_blur.mp4 # Motion blur
    │
    ├── scaling/         # Resolution degradation
    │   ├── *_scaled_480p.mp4 # Downscaled to 480p and back
    │   └── *_scaled_360p.mp4 # Downscaled to 360p and back
    │
    ├── temporal/        # Temporal artifacts
    │   ├── *_framedrop.mp4  # Frame dropping (15fps)
    │   └── *_stutter.mp4    # Frame stutter
    │
    ├── color/           # Color-related issues
    │   ├── *_banding.mp4    # Color banding
    │   └── *_color_shift.mp4 # Hue/saturation shift
    │
    ├── combined/        # Multiple degradations
    │   ├── *_mobile.mp4     # Typical mobile quality
    │   └── *_poor.mp4       # Poor quality combination
    │
    └── degradation_metadata.json  # Detailed degradation info

```

## Usage

### Download Original Videos
```bash
cd scripts
python download_test_videos.py
```

This will download:
- Big Buck Bunny (1080p, 720p, 480p)
- Tears of Steel (1080p, 720p)
- Sintel (1080p, 720p)

### Create Degraded Versions
```bash
cd scripts
python create_degraded_videos.py
```

This creates degraded versions with:
- **Compression artifacts**: Different CRF and bitrate settings
- **Noise**: Gaussian and salt-and-pepper noise
- **Blur**: Box blur and motion blur
- **Scaling**: Downscale and upscale artifacts
- **Temporal**: Frame drops and stutter
- **Color**: Banding and color shifts
- **Combined**: Multiple degradations

## Expected Quality Scores

### Original Videos
- Should score 0.7-0.9 on most metrics
- Professional quality content

### Degraded Videos by Category

#### Compression
- `_crf40`: Medium quality (0.4-0.6)
- `_crf50`: Poor quality (0.2-0.4)
- `_500kbps`: Low quality (0.3-0.5)
- `_200kbps`: Very poor (0.1-0.3)

#### Noise
- `_noise_low`: Slightly degraded (0.5-0.7)
- `_noise_high`: Significantly degraded (0.3-0.5)
- `_salt_pepper`: Noticeably degraded (0.4-0.6)

#### Blur
- `_blur_low`: Mildly degraded (0.5-0.7)
- `_blur_high`: Heavily degraded (0.2-0.4)
- `_motion_blur`: Variable (0.3-0.6)

#### Other Categories
- Scaling: 0.4-0.6 (resolution loss)
- Temporal: 0.3-0.6 (motion artifacts)
- Color: 0.4-0.7 (banding detector specific)
- Combined: 0.2-0.5 (multiple issues)

## Testing Metrics

Use these videos to test:
1. **Sensitivity**: Can metrics detect quality differences?
2. **Ranking**: Do metrics rank videos correctly?
3. **Artifact detection**: Do specialized metrics detect their target artifacts?
4. **Consistency**: Are scores consistent across runs?
5. **Speed**: Processing time for different resolutions

## Notes

- Videos are in H.264/MP4 format for compatibility
- Audio is preserved but not degraded
- Degradations are cumulative (not applied to already degraded videos)
- Source videos are open-source content (Creative Commons)