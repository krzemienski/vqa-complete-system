# Base CUDA 12 image for GPU-enabled VQA metrics
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="VQA System"
LABEL description="Base CUDA 12 image with Python 3.9 and GPU support for VQA metrics"

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python3.9 \
    python3.9-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Video processing
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    # Image processing
    libopencv-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Scientific computing
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    # CUDA-specific
    cuda-toolkit-12-1 \
    libcudnn8 \
    libcudnn8-dev \
    # Other utilities
    vim \
    htop \
    nvidia-htop \
    && rm -rf /var/lib/apt/lists/*

# Create Python symlinks
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install common Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.9.3 \
    pandas==1.5.3 \
    scikit-learn==1.2.2 \
    scikit-image==0.20.0 \
    opencv-python==4.8.1.78 \
    matplotlib==3.7.1 \
    seaborn==0.12.2 \
    Pillow==9.5.0 \
    tqdm==4.65.0 \
    pyyaml==6.0 \
    requests==2.28.2

# Install video processing libraries
RUN pip install --no-cache-dir \
    av==10.0.0 \
    decord==0.6.0 \
    imageio==2.31.1 \
    imageio-ffmpeg==0.4.8

# Install PyTorch with CUDA 12 support
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install additional GPU-accelerated libraries
RUN pip install --no-cache-dir \
    cupy-cuda12x==12.2.0 \
    pycuda==2022.2.2

# Install deep learning frameworks extensions
RUN pip install --no-cache-dir \
    timm==0.9.2 \
    einops==0.7.0 \
    transformers==4.35.2 \
    accelerate==0.24.1

# Set up working directory
WORKDIR /app

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/output

# Set environment variables for CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Enable GPU support in Docker
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Add a health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print(f'CUDA OK: {torch.cuda.get_device_name(0)}')" || exit 1

# Default command to show CUDA info
CMD ["python", "-c", "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"]