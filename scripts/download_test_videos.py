#!/usr/bin/env python3
"""
Download test videos for VQA metric testing.
Downloads Big Buck Bunny, Tears of Steel, and Sintel in multiple resolutions.
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Video sources with multiple resolutions
VIDEO_SOURCES = {
    "big_buck_bunny": {
        "1080p": {
            "url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
            "filename": "bbb_1080p.mp4",
            "md5": None  # Add checksums if available
        },
        "720p": {
            "url": "http://mirrors.standaloneinstaller.com/video-sample/big_buck_bunny_720p_5mb.mp4",
            "filename": "bbb_720p.mp4",
            "md5": None
        },
        "480p": {
            "url": "http://mirrors.standaloneinstaller.com/video-sample/big_buck_bunny_480p_1mb.mp4",
            "filename": "bbb_480p.mp4",
            "md5": None
        }
    },
    "tears_of_steel": {
        "1080p": {
            "url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
            "filename": "tos_1080p.mp4",
            "md5": None
        },
        "720p": {
            "url": "http://mirrors.standaloneinstaller.com/video-sample/tears_of_steel_720p.mp4",
            "filename": "tos_720p.mp4",
            "md5": None
        }
    },
    "sintel": {
        "1080p": {
            "url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4",
            "filename": "sintel_1080p.mp4",
            "md5": None
        },
        "720p": {
            "url": "http://mirrors.standaloneinstaller.com/video-sample/sintel_720p.mp4",
            "filename": "sintel_720p.mp4",
            "md5": None
        }
    }
}

# Alternative sources if primary fails
ALTERNATIVE_SOURCES = {
    "xiph": {
        "bbb_1080p": "https://media.xiph.org/video/derf/y4m/big_buck_bunny_1080p24.y4m",
        "sintel_1080p": "https://media.xiph.org/video/derf/y4m/sintel_trailer_2k_1080p24.y4m"
    },
    "blender": {
        "bbb_1080p": "https://download.blender.org/demo/movies/BBB/bbb_sunflower_1080p_30fps_normal.mp4",
        "tos_1080p": "https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mp4",
        "sintel_1080p": "https://download.blender.org/demo/movies/Sintel/sintel_trailer-1080p.mp4"
    }
}


def download_with_progress(url: str, filepath: Path) -> bool:
    """Download file with progress indication."""
    try:
        def download_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
            sys.stdout.write(f'\rDownloading: {percent:.1f}%')
            sys.stdout.flush()
        
        print(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, filepath, reporthook=download_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def verify_checksum(filepath: Path, expected_md5: str) -> bool:
    """Verify file checksum if provided."""
    if not expected_md5:
        return True
    
    print(f"Verifying checksum...")
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    
    actual_md5 = md5_hash.hexdigest()
    if actual_md5 == expected_md5:
        print("✓ Checksum verified")
        return True
    else:
        print(f"✗ Checksum mismatch: expected {expected_md5}, got {actual_md5}")
        return False


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_video_format(input_path: Path, output_path: Path) -> bool:
    """Convert video to standard format if needed."""
    try:
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y", str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e}")
        return False


def main():
    # Create videos directory
    videos_dir = Path("../test_videos/original")
    videos_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for ffmpeg
    if not check_ffmpeg():
        print("Warning: ffmpeg not found. Some video conversions may fail.")
    
    # Download statistics
    total_videos = sum(len(resolutions) for resolutions in VIDEO_SOURCES.values())
    downloaded = 0
    failed = []
    
    print(f"Downloading {total_videos} test videos...\n")
    
    # Download each video
    for video_name, resolutions in VIDEO_SOURCES.items():
        print(f"\n{'='*50}")
        print(f"Downloading {video_name.replace('_', ' ').title()}")
        print(f"{'='*50}")
        
        for resolution, info in resolutions.items():
            output_path = videos_dir / info["filename"]
            
            # Skip if already exists
            if output_path.exists():
                print(f"✓ {info['filename']} already exists, skipping...")
                downloaded += 1
                continue
            
            # Try primary source
            print(f"\n{resolution}: ", end="")
            success = download_with_progress(info["url"], output_path)
            
            # Try alternative sources if primary fails
            if not success:
                print("Trying alternative sources...")
                alt_key = f"{video_name[:3]}_{resolution}"
                for source_name, alt_urls in ALTERNATIVE_SOURCES.items():
                    if alt_key in alt_urls:
                        success = download_with_progress(alt_urls[alt_key], output_path)
                        if success:
                            break
            
            # Verify checksum if successful
            if success and info.get("md5"):
                if not verify_checksum(output_path, info["md5"]):
                    output_path.unlink()
                    success = False
            
            # Update statistics
            if success:
                downloaded += 1
                print(f"✓ Successfully downloaded {info['filename']}")
                
                # Convert Y4M to MP4 if needed
                if str(output_path).endswith('.y4m'):
                    mp4_path = output_path.with_suffix('.mp4')
                    print(f"Converting Y4M to MP4...")
                    if convert_video_format(output_path, mp4_path):
                        output_path.unlink()
                        print(f"✓ Converted to {mp4_path.name}")
                    else:
                        print(f"✗ Conversion failed, keeping Y4M format")
            else:
                failed.append(f"{video_name} - {resolution}")
                print(f"✗ Failed to download {info['filename']}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Download Summary")
    print(f"{'='*50}")
    print(f"Total videos: {total_videos}")
    print(f"Downloaded: {downloaded}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print(f"\nFailed downloads:")
        for item in failed:
            print(f"  - {item}")
    
    if downloaded > 0:
        print(f"\nVideos saved to: {videos_dir.absolute()}")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)