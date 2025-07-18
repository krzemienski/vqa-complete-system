#!/usr/bin/env python3
"""
Create degraded versions of test videos for VQA metric testing.
Applies various types of degradations: compression, noise, blur, etc.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import json
import time

# Degradation configurations
DEGRADATIONS = {
    "compression": [
        {
            "name": "high_compression",
            "description": "High H.264 compression (CRF 40)",
            "suffix": "_crf40",
            "command": ["-c:v", "libx264", "-crf", "40", "-preset", "fast"]
        },
        {
            "name": "very_high_compression",
            "description": "Very high H.264 compression (CRF 50)",
            "suffix": "_crf50",
            "command": ["-c:v", "libx264", "-crf", "50", "-preset", "fast"]
        },
        {
            "name": "low_bitrate",
            "description": "Low bitrate encoding (500kbps)",
            "suffix": "_500kbps",
            "command": ["-c:v", "libx264", "-b:v", "500k", "-maxrate", "500k", "-bufsize", "1M"]
        },
        {
            "name": "very_low_bitrate",
            "description": "Very low bitrate encoding (200kbps)",
            "suffix": "_200kbps",
            "command": ["-c:v", "libx264", "-b:v", "200k", "-maxrate", "200k", "-bufsize", "500k"]
        }
    ],
    "noise": [
        {
            "name": "gaussian_noise_low",
            "description": "Low Gaussian noise",
            "suffix": "_noise_low",
            "command": ["-vf", "noise=alls=10:allf=t+u"]
        },
        {
            "name": "gaussian_noise_high",
            "description": "High Gaussian noise",
            "suffix": "_noise_high",
            "command": ["-vf", "noise=alls=30:allf=t+u"]
        },
        {
            "name": "salt_pepper",
            "description": "Salt and pepper noise",
            "suffix": "_salt_pepper",
            "command": ["-vf", "noise=alls=20:allf=p"]
        }
    ],
    "blur": [
        {
            "name": "blur_low",
            "description": "Low blur (radius 2)",
            "suffix": "_blur_low",
            "command": ["-vf", "boxblur=2:2"]
        },
        {
            "name": "blur_high",
            "description": "High blur (radius 5)",
            "suffix": "_blur_high",
            "command": ["-vf", "boxblur=5:5"]
        },
        {
            "name": "motion_blur",
            "description": "Motion blur simulation",
            "suffix": "_motion_blur",
            "command": ["-vf", "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:vsbmc=1,tblend=all_mode=average"]
        }
    ],
    "scaling": [
        {
            "name": "downscale_480p",
            "description": "Downscale to 480p and back",
            "suffix": "_scaled_480p",
            "command": ["-vf", "scale=854:480:flags=bilinear,scale=1920:1080:flags=bilinear"]
        },
        {
            "name": "downscale_360p",
            "description": "Downscale to 360p and back",
            "suffix": "_scaled_360p",
            "command": ["-vf", "scale=640:360:flags=bilinear,scale=1920:1080:flags=bilinear"]
        }
    ],
    "temporal": [
        {
            "name": "frame_drop",
            "description": "Drop frames (simulate 15fps)",
            "suffix": "_framedrop",
            "command": ["-vf", "select='not(mod(n\\,2))',setpts=0.5*PTS", "-r", "15"]
        },
        {
            "name": "stutter",
            "description": "Add frame stutter",
            "suffix": "_stutter",
            "command": ["-vf", "telecine=pattern=5,dejudder"]
        }
    ],
    "color": [
        {
            "name": "banding",
            "description": "Color banding (8-bit quantization)",
            "suffix": "_banding",
            "command": ["-vf", "format=yuv420p,eq=contrast=1.2", "-pix_fmt", "yuv420p", "-sws_dither", "none"]
        },
        {
            "name": "color_shift",
            "description": "Color shift/saturation issues",
            "suffix": "_color_shift",
            "command": ["-vf", "hue=h=30:s=0.7"]
        }
    ],
    "combined": [
        {
            "name": "mobile_quality",
            "description": "Typical mobile upload quality",
            "suffix": "_mobile",
            "command": ["-vf", "scale=854:480,noise=alls=5:allf=t", "-c:v", "libx264", "-crf", "30", "-preset", "fast"]
        },
        {
            "name": "poor_quality",
            "description": "Multiple degradations combined",
            "suffix": "_poor",
            "command": ["-vf", "scale=640:360,boxblur=1:1,noise=alls=10:allf=t+u,scale=1920:1080", "-c:v", "libx264", "-crf", "35"]
        }
    ]
}


def run_ffmpeg_command(input_path: Path, output_path: Path, ffmpeg_args: List[str]) -> Tuple[bool, str]:
    """Run ffmpeg command and return success status and any error message."""
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-y",  # Overwrite output
        *ffmpeg_args,
        "-c:a", "copy",  # Copy audio as-is
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, f"FFmpeg error: {e.stderr}"


def check_video_integrity(video_path: Path) -> bool:
    """Check if video file is valid and can be processed."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-of", "json",
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) > 0
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False


def create_degraded_video(input_path: Path, output_dir: Path, degradation: Dict) -> bool:
    """Create a single degraded version of a video."""
    # Generate output filename
    output_name = input_path.stem + degradation["suffix"] + input_path.suffix
    output_path = output_dir / output_name
    
    # Skip if already exists
    if output_path.exists():
        print(f"  ✓ {output_name} already exists, skipping...")
        return True
    
    print(f"  Creating {output_name}: {degradation['description']}...")
    
    # Apply degradation
    success, error = run_ffmpeg_command(input_path, output_path, degradation["command"])
    
    if success:
        # Verify output
        if check_video_integrity(output_path):
            print(f"    ✓ Successfully created {output_name}")
            return True
        else:
            print(f"    ✗ Output file corrupted, removing...")
            output_path.unlink(missing_ok=True)
            return False
    else:
        print(f"    ✗ Failed: {error}")
        return False


def create_metadata_file(output_dir: Path, results: Dict):
    """Create metadata file with degradation information."""
    metadata_path = output_dir / "degradation_metadata.json"
    
    metadata = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "degradations": DEGRADATIONS,
        "results": results
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_path}")


def main():
    # Check for ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg and ffprobe must be installed")
        return False
    
    # Setup directories
    input_dir = Path("../test_videos/original")
    output_base = Path("../test_videos/degraded")
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        print("Please run download_test_videos.py first")
        return False
    
    # Find all video files
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
    if not video_files:
        print(f"No video files found in {input_dir}")
        return False
    
    print(f"Found {len(video_files)} source videos")
    print(f"Will create {len(video_files) * sum(len(cat) for cat in DEGRADATIONS.values())} degraded versions\n")
    
    # Process each video
    results = {}
    total_created = 0
    total_failed = 0
    
    for video_path in sorted(video_files):
        print(f"\n{'='*60}")
        print(f"Processing: {video_path.name}")
        print(f"{'='*60}")
        
        # Check source video
        if not check_video_integrity(video_path):
            print(f"✗ Source video appears corrupted, skipping...")
            continue
        
        results[video_path.name] = {}
        
        # Apply each category of degradation
        for category, degradation_list in DEGRADATIONS.items():
            output_dir = output_base / category
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{category.upper()} degradations:")
            results[video_path.name][category] = {}
            
            for degradation in degradation_list:
                success = create_degraded_video(video_path, output_dir, degradation)
                results[video_path.name][category][degradation["name"]] = success
                
                if success:
                    total_created += 1
                else:
                    total_failed += 1
    
    # Create metadata file
    create_metadata_file(output_base, results)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Degradation Summary")
    print(f"{'='*60}")
    print(f"Total videos created: {total_created}")
    print(f"Failed: {total_failed}")
    print(f"\nDegraded videos saved to: {output_base.absolute()}")
    
    # Print category summary
    print(f"\nBy category:")
    for category in DEGRADATIONS:
        category_dir = output_base / category
        if category_dir.exists():
            count = len(list(category_dir.glob("*.mp4")))
            print(f"  {category}: {count} videos")
    
    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)