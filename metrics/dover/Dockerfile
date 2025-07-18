# DOVER (Disentangled Objective Video Quality Evaluator) Docker Image
FROM vqa-system/python39-base:latest

LABEL maintainer="VQA System"
LABEL description="DOVER - Disentangled video quality assessment (technical + aesthetic)"

# Install DOVER-specific dependencies (compatible with base image)
RUN pip install --no-cache-dir \
    timm==0.9.2 \
    einops==0.7.0 \
    pytorchvideo==0.1.5 \
    wandb==0.15.12 \
    gdown==4.7.1

# Create directories
WORKDIR /app
RUN mkdir -p /app/models /app/src /app/data

# Copy DOVER implementation
COPY src/ /app/src/
COPY run_dover.py /app/

# Download model weights
RUN python -c "import gdown; \
    gdown.download('https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth', '/app/models/DOVER.pth', quiet=False); \
    gdown.download('https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER-Mobile.pth', '/app/models/DOVER-Mobile.pth', quiet=False)" || \
    echo "Model download failed, will retry at runtime"

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV DOVER_MODEL_PATH=/app/models

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; from src.dover import DOVER; print('DOVER OK')" || exit 1

# Default command
ENTRYPOINT ["python", "/app/run_dover.py"]
CMD ["--help"]