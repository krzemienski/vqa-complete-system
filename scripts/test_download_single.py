#!/usr/bin/env python3
"""
Test script to download a single small video to verify setup.
"""

import urllib.request
import sys
from pathlib import Path

def download_test_video():
    """Download a single small test video."""
    # Try multiple sources
    test_urls = [
        ("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4", "test_blazes.mp4"),
        ("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerEscapes.mp4", "test_escapes.mp4"),
        ("https://sample-videos.com/video321/mp4/480/big_buck_bunny_480p_1mb.mp4", "test_bbb_480p.mp4"),
        ("https://www.w3schools.com/html/mov_bbb.mp4", "test_bbb_small.mp4")
    ]
    
    output_dir = Path("../test_videos/original")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for url, filename in test_urls:
        output_path = output_dir / filename
        
        if output_path.exists():
            print(f"Test video already exists at: {output_path}")
            return True
        
        print(f"\nTrying to download from: {url}")
        
        try:
            def download_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
                sys.stdout.write(f'\rProgress: {percent:.1f}%')
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, output_path, reporthook=download_hook)
            print(f"\n✓ Successfully downloaded test video: {filename}")
            print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
            print(f"  Saved to: {output_path.absolute()}")
            return True
            
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            if output_path.exists():
                output_path.unlink()
            continue
    
    print("\n✗ All download attempts failed")
    return False

if __name__ == "__main__":
    success = download_test_video()
    sys.exit(0 if success else 1)