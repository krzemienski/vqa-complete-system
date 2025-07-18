#!/usr/bin/env python3
"""
MDTVSFA metric runner script.
Mixed dataset training for cross-dataset robust video quality assessment.
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
from src.mdtvsfa import MDTVSFA, MDTVSFALite
from src.data_utils import load_video, get_dataset_id, DATASET_IDS


def parse_args():
    parser = argparse.ArgumentParser(description='MDTVSFA Video Quality Assessment')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'lite'],
                        help='Model variant to use (default: full)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: stdout)')
    parser.add_argument('--num_frames', type=int, default=32,
                        help='Number of frames to sample (default: 32)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specify dataset for calibration (konvid1k, youtube_ugc, etc.)')
    return parser.parse_args()


def load_model(model_type='full', device='cpu'):
    """Load MDTVSFA model and weights."""
    model_dir = Path(os.environ.get('MDTVSFA_MODEL_PATH', '/app/models'))
    
    if model_type == 'lite':
        model = MDTVSFALite(num_datasets=len(DATASET_IDS))
        weight_file = 'MDTVSFA-Lite.pth'
    else:
        model = MDTVSFA(num_datasets=len(DATASET_IDS))
        weight_file = 'MDTVSFA.pth'
    
    weight_path = model_dir / weight_file
    
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
            print("Using random initialization for MDTVSFA")
    else:
        print(f"Warning: Model weights not found at {weight_path}, using random initialization")
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_video(video_path, model, num_frames=32, device='cpu', dataset_id=None):
    """Evaluate a single video."""
    # Load video
    try:
        video_data = load_video(video_path, num_frames=num_frames)
        # Ensure float32 data type to match model expectations
        video_tensor = torch.from_numpy(video_data).float().unsqueeze(0).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load video: {e}")
    
    # Get dataset ID if not specified
    if dataset_id is None:
        dataset_id = get_dataset_id(video_path)
    
    # Run inference
    with torch.no_grad():
        if isinstance(model, MDTVSFA):
            # Full model can return features
            score, features = model(video_tensor, dataset_id=dataset_id, return_features=True)
            
            # Extract additional info
            result = {
                'score': float(score.cpu().numpy()[0]),
                'dataset_id': dataset_id,
                'spatial_features_mean': float(features['spatial'].mean().cpu().numpy()),
                'motion_features_mean': float(features['motion'].mean().cpu().numpy())
            }
        else:
            # Lite model only returns score
            score = model(video_tensor)
            result = {
                'score': float(score.cpu().numpy()[0]),
                'dataset_id': dataset_id
            }
    
    # Ensure score is in [0, 1] range
    result['score'] = np.clip(result['score'], 0, 1)
    
    return result


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


def main():
    args = parse_args()
    
    # Validate input
    if not os.path.exists(args.video_path):
        error_result = {
            'status': 'error',
            'error': f"Video file not found: {args.video_path}",
            'metric': 'MDTVSFA'
        }
        print(json.dumps(error_result))
        sys.exit(1)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Load model
        model = load_model(model_type=args.model, device=args.device)
        
        # Get dataset ID if specified
        dataset_id = None
        if args.dataset:
            dataset_id = DATASET_IDS.get(args.dataset.lower())
            if dataset_id is None:
                print(f"Warning: Unknown dataset '{args.dataset}', using generic", file=sys.stderr)
                dataset_id = DATASET_IDS['generic']
        
        # Evaluate video
        eval_result = evaluate_video(
            args.video_path, 
            model, 
            num_frames=args.num_frames,
            device=args.device,
            dataset_id=dataset_id
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        video_duration = get_video_duration(args.video_path)
        speed_factor = video_duration / execution_time if execution_time > 0 else 0
        
        # Prepare result
        result = {
            'status': 'success',
            'metric': 'MDTVSFA' if args.model == 'full' else 'MDTVSFA-Lite',
            'video': os.path.basename(args.video_path),
            'score': eval_result['score'],
            'dataset_id': eval_result['dataset_id'],
            'dataset_name': [k for k, v in DATASET_IDS.items() if v == eval_result['dataset_id']][0],
            'num_frames': args.num_frames,
            'execution_time': execution_time,
            'speed_factor': speed_factor,
            'device': args.device
        }
        
        # Add feature info if available
        if 'spatial_features_mean' in eval_result:
            result['spatial_features_mean'] = eval_result['spatial_features_mean']
            result['motion_features_mean'] = eval_result['motion_features_mean']
        
    except Exception as e:
        # Error handling
        result = {
            'status': 'error',
            'metric': 'MDTVSFA' if args.model == 'full' else 'MDTVSFA-Lite',
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


if __name__ == '__main__':
    main()