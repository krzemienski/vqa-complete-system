# Base Python 3.9 image for VQA metrics
FROM python:3.9-slim-bullseye

LABEL maintainer="VQA System"
LABEL description="Base Python 3.9 image with common dependencies for VQA metrics"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
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
    # Other utilities
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install common Python packages
RUN pip install --upgrade pip setuptools wheel

# Install common Python dependencies for VQA
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    scipy==1.9.1 \
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
    imageio==2.31.1 \
    imageio-ffmpeg==0.4.8 \
    scikit-video==1.1.11
    
# Try to install decord (may not be available on all architectures)
RUN pip install --no-cache-dir decord==0.6.0 || echo "Decord not available for this architecture"

# Install PyTorch dependencies (for ARM64 compatibility)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2

# Set up working directory
WORKDIR /app

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/output

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Add a health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import cv2, numpy, scipy, torch; print('OK')" || exit 1

# Default command
CMD ["python", "--version"]