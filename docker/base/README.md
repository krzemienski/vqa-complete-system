# Base Docker Images for VQA System

This directory contains base Docker images that serve as foundations for all VQA metric containers.

## Available Base Images

### 1. Python 3.8 Base (`python38.Dockerfile`)
- **Image**: `vqa-system/python38-base:latest`
- **Purpose**: For metrics requiring Python 3.8 compatibility
- **Includes**: 
  - Python 3.8 with scientific computing stack
  - OpenCV, FFmpeg, video processing libraries
  - Common VQA dependencies
- **Size**: ~1.5GB

### 2. Python 3.9 Base (`python39.Dockerfile`)
- **Image**: `vqa-system/python39-base:latest`
- **Purpose**: For modern metrics using Python 3.9+
- **Includes**:
  - Python 3.9 with latest package versions
  - PyTorch CPU support
  - Deep learning frameworks
- **Size**: ~2GB

### 3. CUDA 12 Base (`cuda12.Dockerfile`)
- **Image**: `vqa-system/cuda12-base:latest`
- **Purpose**: For GPU-accelerated metrics
- **Includes**:
  - CUDA 12.1 with cuDNN 8
  - PyTorch with CUDA support
  - GPU-accelerated libraries
- **Requirements**: NVIDIA GPU with driver 525.60+
- **Size**: ~6GB

### 4. MATLAB Runtime Base (`matlab.Dockerfile`)
- **Image**: `vqa-system/matlab-base:latest`
- **Purpose**: For MATLAB-based metrics (RAPIQUE, VIDEVAL, etc.)
- **Includes**:
  - MATLAB Runtime R2023b (9.13)
  - Python wrapper support
  - Video processing tools
- **Size**: ~3GB

## Building Base Images

### Build All Images
```bash
./build_base_images.sh all
```

### Build Specific Image
```bash
./build_base_images.sh python38  # Python 3.8 base
./build_base_images.sh python39  # Python 3.9 base
./build_base_images.sh cuda12    # CUDA 12 base
./build_base_images.sh matlab    # MATLAB Runtime base
```

### Using Docker Compose
```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build python38-base
```

## Testing Base Images

### Test Python 3.8
```bash
docker run --rm vqa-system/python38-base:latest python -c "import cv2, numpy; print('Python 3.8 OK')"
```

### Test Python 3.9
```bash
docker run --rm vqa-system/python39-base:latest python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Test CUDA 12
```bash
docker run --rm --gpus all vqa-system/cuda12-base:latest
```

### Test MATLAB Runtime
```bash
docker run --rm vqa-system/matlab-base:latest ls /usr/local/MATLAB/MATLAB_Runtime/
```

## Usage in Metric Dockerfiles

Example of using a base image in a metric Dockerfile:

```dockerfile
# For Python-based metric
FROM vqa-system/python39-base:latest

# For GPU-accelerated metric
FROM vqa-system/cuda12-base:latest

# For MATLAB-based metric
FROM vqa-system/matlab-base:latest
```

## Optimization Tips

1. **Layer Caching**: Base images are designed to maximize Docker layer caching
2. **Multi-stage Builds**: Use these as base for multi-stage builds
3. **Size Optimization**: Choose the minimal base for your metric's needs
4. **GPU Support**: Only use CUDA base if GPU acceleration is required

## Troubleshooting

### NVIDIA Docker Issues
If CUDA image fails to build or run:
```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### MATLAB Runtime Download
If MATLAB Runtime download fails:
1. Check the URL in `matlab.Dockerfile` is still valid
2. Download manually and modify Dockerfile to use local file
3. Consider using pre-built image from registry

### Space Issues
Base images are large. Ensure sufficient disk space:
```bash
# Check Docker space usage
docker system df

# Clean up unused images
docker system prune -a
```

## Maintenance

### Updating Base Images
1. Update package versions in Dockerfiles
2. Rebuild affected base images
3. Test all dependent metric containers
4. Update documentation

### Security Updates
```bash
# Rebuild with latest security patches
docker build --no-cache -f python39.Dockerfile -t vqa-system/python39-base:latest .
```