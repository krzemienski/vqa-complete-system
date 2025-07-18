#!/usr/bin/env python3
"""
Test script to create a single degraded video to verify degradation functionality.
"""

import subprocess
from pathlib import Path
import sys

def test_single_degradation():
    """Test creating one degraded version of the test video."""
    # Input and output paths
    input_path = Path("../test_videos/original/test_blazes.mp4")
    output_dir = Path("../test_videos/degraded/compression")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Test video not found at {input_path}")
        print("Please run test_download_single.py first")
        return False
    
    output_path = output_dir / "test_blazes_crf40.mp4"
    
    # Test degradation: high compression
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-y",  # Overwrite
        "-c:v", "libx264",
        "-crf", "40",
        "-preset", "fast",
        "-c:a", "copy",
        str(output_path)
    ]
    
    print(f"Testing video degradation...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Degradation: H.264 CRF 40 (high compression)")
    
    try:
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Check output file
            if output_path.exists() and output_path.stat().st_size > 0:
                input_size = input_path.stat().st_size / 1024 / 1024
                output_size = output_path.stat().st_size / 1024 / 1024
                compression = (1 - output_size / input_size) * 100
                
                print(f"\n✓ Successfully created degraded video")
                print(f"  Original size: {input_size:.2f} MB")
                print(f"  Degraded size: {output_size:.2f} MB")
                print(f"  Compression: {compression:.1f}%")
                print(f"  Saved to: {output_path.absolute()}")
                return True
            else:
                print(f"\n✗ Output file is empty or missing")
                return False
        else:
            print(f"\n✗ FFmpeg error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_single_degradation()
    sys.exit(0 if success else 1)