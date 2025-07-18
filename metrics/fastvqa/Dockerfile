# Fast-VQA / FasterVQA Docker Image
FROM vqa-system/python39-base:latest

LABEL maintainer="VQA System"
LABEL description="Fast-VQA/FasterVQA - Real-time video quality assessment"

# Install Fast-VQA specific dependencies (compatible with base image)
RUN pip install --no-cache-dir \
    timm==0.9.2 \
    einops==0.7.0 \
    pytorchvideo==0.1.5 \
    gdown==4.7.1

# Create directories
WORKDIR /app
RUN mkdir -p /app/models /app/src /app/data

# Copy implementation
COPY src/ /app/src/
COPY run_fastvqa.py /app/

# Download model weights
RUN python -c "import gdown; \
    gdown.download('https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST-VQA-B-HV.pth', '/app/models/FAST-VQA-B.pth', quiet=False); \
    gdown.download('https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FasterVQA.pth', '/app/models/FasterVQA.pth', quiet=False)" || \
    echo "Model download failed, will retry at runtime"

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH
ENV FASTVQA_MODEL_PATH=/app/models

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch; from src.fastvqa import FastVQA; print('FastVQA OK')" || exit 1

# Default command
ENTRYPOINT ["python", "/app/run_fastvqa.py"]
CMD ["--help"]