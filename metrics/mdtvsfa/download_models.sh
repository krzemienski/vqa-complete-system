#!/bin/bash
#
# Download MDTVSFA model weights

set -e

# Create models directory
mkdir -p models

echo "Downloading MDTVSFA model weights..."

# MDTVSFA Full model
if [ ! -f "models/MDTVSFA.pth" ]; then
    echo "Downloading MDTVSFA full model..."
    # Using placeholder URLs - replace with actual model URLs
    wget -O models/MDTVSFA.pth \
        "https://github.com/VQAssessment/MDTVSFA/releases/download/v1.0/MDTVSFA.pth" || \
    curl -L -o models/MDTVSFA.pth \
        "https://github.com/VQAssessment/MDTVSFA/releases/download/v1.0/MDTVSFA.pth" || \
    echo "Warning: Could not download MDTVSFA full model"
else
    echo "MDTVSFA full model already exists"
fi

# MDTVSFA Lite model
if [ ! -f "models/MDTVSFA-Lite.pth" ]; then
    echo "Downloading MDTVSFA lite model..."
    wget -O models/MDTVSFA-Lite.pth \
        "https://github.com/VQAssessment/MDTVSFA/releases/download/v1.0/MDTVSFA-Lite.pth" || \
    curl -L -o models/MDTVSFA-Lite.pth \
        "https://github.com/VQAssessment/MDTVSFA/releases/download/v1.0/MDTVSFA-Lite.pth" || \
    echo "Warning: Could not download MDTVSFA lite model"
else
    echo "MDTVSFA lite model already exists"
fi

echo "Model download complete!"
ls -lh models/