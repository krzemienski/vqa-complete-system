#!/bin/bash
#
# Complete TODO 2: Download and prepare all test videos
# This script runs both download and degradation processes

echo "=================================================="
echo "TODO 2: Download and Prepare Test Videos"
echo "=================================================="
echo ""

# Check for dependencies
echo "Checking dependencies..."
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is required but not installed"
    echo "Install with: brew install ffmpeg"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed"
    exit 1
fi

echo "✓ Dependencies verified"
echo ""

# Step 1: Download original videos
echo "Step 1: Downloading original test videos..."
echo "------------------------------------------"
python3 download_test_videos.py

if [ $? -ne 0 ]; then
    echo "Error: Download failed"
    exit 1
fi

echo ""
echo "✓ Download completed"
echo ""

# Step 2: Create degraded versions
echo "Step 2: Creating degraded video versions..."
echo "------------------------------------------"
python3 create_degraded_videos.py

if [ $? -ne 0 ]; then
    echo "Error: Degradation failed"
    exit 1
fi

echo ""
echo "✓ Degradation completed"
echo ""

# Summary
echo "=================================================="
echo "TODO 2 COMPLETED SUCCESSFULLY!"
echo "=================================================="
echo ""
echo "Results:"
echo "  Original videos: $(find ../test_videos/original -name "*.mp4" | wc -l)"
echo "  Degraded videos: $(find ../test_videos/degraded -name "*.mp4" | wc -l)"
echo ""
echo "Total disk usage:"
du -sh ../test_videos/
echo ""
echo "Next step: TODO 3 - Build Base Docker Images"