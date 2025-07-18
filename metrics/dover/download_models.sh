#!/bin/bash
#
# Download DOVER model weights

set -e

# Create models directory
mkdir -p models

echo "Downloading DOVER model weights..."

# DOVER Full model
if [ ! -f "models/DOVER.pth" ]; then
    echo "Downloading DOVER full model..."
    # Using wget as fallback since gdown might have issues
    wget -O models/DOVER.pth \
        "https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth" || \
    curl -L -o models/DOVER.pth \
        "https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth"
else
    echo "DOVER full model already exists"
fi

# DOVER Mobile model
if [ ! -f "models/DOVER-Mobile.pth" ]; then
    echo "Downloading DOVER mobile model..."
    wget -O models/DOVER-Mobile.pth \
        "https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER-Mobile.pth" || \
    curl -L -o models/DOVER-Mobile.pth \
        "https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER-Mobile.pth"
else
    echo "DOVER mobile model already exists"
fi

echo "Model download complete!"
ls -lh models/