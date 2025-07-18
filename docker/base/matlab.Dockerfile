# Base MATLAB image for MATLAB-based VQA metrics
# Note: Requires MATLAB license and runtime installer
FROM ubuntu:20.04

LABEL maintainer="VQA System"
LABEL description="Base MATLAB Runtime image for MATLAB-based VQA metrics"

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    wget \
    curl \
    unzip \
    ca-certificates \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    # Python for wrapper scripts
    python3.8 \
    python3-pip \
    # Video processing
    ffmpeg \
    # Required by MATLAB Runtime
    libglib2.0-0 \
    libgomp1 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libxcomposite1 \
    libasound2 \
    libatk1.0-0 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libgcc1 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libxcursor1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libxss1 \
    && rm -rf /var/lib/apt/lists/*

# MATLAB Runtime version R2023b (9.13)
ENV MCR_VERSION=R2023b
ENV MCR_RELEASE=9_13

# Download and install MATLAB Runtime
# Note: This URL may need updating based on your MATLAB version
RUN mkdir /tmp/mcr_install && \
    cd /tmp/mcr_install && \
    wget -q https://ssd.mathworks.com/supportfiles/downloads/R2023b/Release/0/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2023b_glnxa64.zip && \
    unzip -q MATLAB_Runtime_R2023b_glnxa64.zip && \
    ./install -mode silent -agreeToLicense yes -destinationFolder /usr/local/MATLAB/MATLAB_Runtime && \
    rm -rf /tmp/mcr_install

# Configure MATLAB Runtime environment
ENV LD_LIBRARY_PATH=/usr/local/MATLAB/MATLAB_Runtime/R2023b/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2023b/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/R2023b/sys/os/glnxa64:$LD_LIBRARY_PATH
ENV XAPPLRESDIR=/usr/local/MATLAB/MATLAB_Runtime/R2023b/X11/app-defaults
ENV MCR_INHIBIT_CTF_LOCK=1

# Install Python dependencies for wrappers
RUN pip3 install --no-cache-dir \
    numpy==1.21.6 \
    scipy==1.7.3 \
    opencv-python==4.8.1.78 \
    pyyaml==6.0

# Create working directory
WORKDIR /app

# Create directories for MATLAB applications and data
RUN mkdir -p /app/matlab_apps /app/models /app/data /app/output

# Create a wrapper script for MATLAB apps
RUN echo '#!/bin/bash\n\
if [ -z "$1" ]; then\n\
    echo "Usage: run_matlab_app.sh <app_path> [arguments]"\n\
    exit 1\n\
fi\n\
APP_PATH=$1\n\
shift\n\
exec "$APP_PATH" "$@"' > /usr/local/bin/run_matlab_app.sh && \
    chmod +x /usr/local/bin/run_matlab_app.sh

# Health check to verify MATLAB Runtime
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD test -d /usr/local/MATLAB/MATLAB_Runtime/${MCR_VERSION} || exit 1

# Default command
CMD ["echo", "MATLAB Runtime R2023b installed. Use run_matlab_app.sh to execute compiled MATLAB applications."]