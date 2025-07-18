#!/bin/bash
#
# Download Fast-VQA and FasterVQA model weights

set -e

# Create models directory
mkdir -p models

echo "Downloading Fast-VQA model weights..."

# Fast-VQA Base model
if [ ! -f "models/FAST-VQA-B.pth" ]; then
    echo "Downloading Fast-VQA Base model..."
    # Try GitHub releases first
    wget -O models/FAST-VQA-B.pth \
        "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST-VQA-B-HV.pth" || \
    curl -L -o models/FAST-VQA-B.pth \
        "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FAST-VQA-B-HV.pth" || \
    echo "Warning: Could not download Fast-VQA-B model"
else
    echo "Fast-VQA Base model already exists"
fi

# FasterVQA model
if [ ! -f "models/FasterVQA.pth" ]; then
    echo "Downloading FasterVQA model..."
    wget -O models/FasterVQA.pth \
        "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FasterVQA.pth" || \
    curl -L -o models/FasterVQA.pth \
        "https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/download/v2.0.0/FasterVQA.pth" || \
    echo "Warning: Could not download FasterVQA model"
else
    echo "FasterVQA model already exists"
fi

echo "Model download complete!"
ls -lh models/