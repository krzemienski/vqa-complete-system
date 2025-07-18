# MDTVSFA Docker Image
FROM vqa-system/python39-base:latest

LABEL maintainer="VQA System"
LABEL description="MDTVSFA - Mixed Dataset Training for cross-dataset robust VQA"

# Install MDTVSFA specific dependencies (compatible with base image)
RUN pip install --no-cache-dir \
    timm==0.9.2 \
    einops==0.7.0 \
    gdown==4.7.1

# Create directories
WORKDIR /app
RUN mkdir -p /app/models /app/src /app/data

# Copy implementation
COPY src/ /app/src/
COPY run_mdtvsfa.py /app/

# Download model weights
RUN python -c "import gdown; \
    gdown.download('https://github.com/VQAssessment/MDTVSFA/releases/download/v1.0/MDTVSFA.pth', '/app/models/MDTVSFA.pth', quiet=False); \
    gdown.download('https://github.com/VQAssessment/MDTVSFA/releases/download/v1.0/MDTVSFA-Lite.pth', '/app/models/MDTVSFA-Lite.pth', quiet=False)" || \
    echo "Model download failed, will retry at runtime"

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV MDTVSFA_MODEL_PATH=/app/models

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; from src.mdtvsfa import MDTVSFA; print('MDTVSFA OK')" || exit 1

# Default command
ENTRYPOINT ["python", "/app/run_mdtvsfa.py"]
CMD ["--help"]