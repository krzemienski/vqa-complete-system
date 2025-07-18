#!/usr/bin/env python3
"""
Fast-VQA / FasterVQA metric runner script.
Real-time video quality assessment with fragment sampling.
"""

import os
import sys
import json
import time
import argparse
import traceback
from pathlib import Path

import torch
import numpy as np
from src.fastvqa import FastVQA, FasterVQA
from src.data_utils import load_video_fragments


def parse_args():
    parser = argparse.ArgumentParser(description='Fast-VQA/FasterVQA Video Quality Assessment')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--model', type=str, default='faster', choices=['fast', 'faster'],
                        help='Model variant to use (default: faster)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: stdout)')
    parser.add_argument('--fragments', type=int, default=None,
                        help='Number of fragments (default: 8 for fast, 4 for faster)')
    return parser.parse_args()


def load_model(model_type='faster', device='cpu'):
    """Load Fast-VQA model and weights."""
    model_dir = Path(os.environ.get('FASTVQA_MODEL_PATH', '/app/models'))
    
    if model_type == 'faster':
        model = FasterVQA()
        weight_path = model_dir / 'FasterVQA.pth'
        default_fragments = 4
    else:
        model = FastVQA()
        weight_path = model_dir / 'FAST-VQA-B.pth'
        default_fragments = 8
    
    # Load weights if available and valid
    if weight_path.exists():
        try:
            # Check if file is valid (not just "Not Found" text)
            if weight_path.stat().st_size > 100:  # Valid models should be much larger than 100 bytes
                checkpoint = torch.load(weight_path, map_location=device)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                    print(f"Loaded weights from {weight_path}")
                else:
                    model.load_state_dict(checkpoint, strict=False)
                    print(f"Loaded weights from {weight_path}")
            else:
                print(f"Warning: Model file {weight_path} is too small (likely download error), using random initialization")
        except Exception as e:
            print(f"Warning: Could not load weights from {weight_path}: {e}")
            print("Using random initialization for Fast-VQA")
    else:
        print(f"Warning: Model weights not found at {weight_path}, using random initialization")
    
    model.to(device)
    model.eval()
    
    return model, default_fragments


def evaluate_video(video_path, model, fragments=4, device='cpu'):
    """Evaluate a single video."""
    # Load video fragments
    try:
        video_data = load_video_fragments(video_path, num_fragments=fragments, frames_per_fragment=8)
        # Ensure float32 data type to match model expectations
        video_tensor = torch.from_numpy(video_data).float().unsqueeze(0).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load video: {e}")
    
    # Run inference
    with torch.no_grad():
        score = model(video_tensor)
    
    # Extract score
    if isinstance(score, torch.Tensor):
        quality_score = float(score.cpu().numpy()[0])
    else:
        quality_score = float(score)
    
    # Ensure score is in [0, 1] range
    quality_score = np.clip(quality_score, 0, 1)
    
    return quality_score


def main():
    args = parse_args()
    
    # Validate input
    if not os.path.exists(args.video_path):
        error_result = {
            'status': 'error',
            'error': f"Video file not found: {args.video_path}",
            'metric': 'Fast-VQA'
        }
        print(json.dumps(error_result))
        sys.exit(1)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Load model
        model, default_fragments = load_model(model_type=args.model, device=args.device)
        fragments = args.fragments if args.fragments is not None else default_fragments
        
        # Evaluate video
        score = evaluate_video(args.video_path, model, fragments=fragments, device=args.device)
        
        # Calculate inference speed
        inference_time = time.time() - start_time
        video_duration = get_video_duration(args.video_path)
        speed_factor = video_duration / inference_time if inference_time > 0 else 0
        
        # Prepare result
        result = {
            'status': 'success',
            'metric': 'Fast-VQA' if args.model == 'fast' else 'FasterVQA',
            'video': os.path.basename(args.video_path),
            'score': score,
            'fragments': fragments,
            'execution_time': inference_time,
            'speed_factor': speed_factor,  # How many times faster than real-time
            'device': args.device
        }
        
    except Exception as e:
        # Error handling
        result = {
            'status': 'error',
            'metric': 'Fast-VQA' if args.model == 'fast' else 'FasterVQA',
            'video': os.path.basename(args.video_path),
            'error': str(e),
            'traceback': traceback.format_exc(),
            'execution_time': time.time() - start_time
        }
    
    # Output results
    output_json = json.dumps(result, indent=2)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output_json)
    else:
        print(output_json)
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] == 'success' else 1)


def get_video_duration(video_path):
    """Get video duration in seconds."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        return frame_count / fps if fps > 0 else 0
    except:
        return 0


if __name__ == '__main__':
    main()