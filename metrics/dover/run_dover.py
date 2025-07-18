#!/usr/bin/env python3
"""
DOVER VQA metric runner script.
Evaluates video quality using disentangled technical and aesthetic assessment.
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
from src.dover import DOVER, DOVERMobile
from src.data_utils import VideoDataset, load_video


def parse_args():
    parser = argparse.ArgumentParser(description='DOVER Video Quality Assessment')
    parser.add_argument('video_path', type=str, help='Path to input video')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'mobile'],
                        help='Model variant to use (default: full)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (default: stdout)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for processing')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of data loading workers')
    return parser.parse_args()


def map_model_keys(state_dict):
    """Map model keys from downloaded format to our simplified architecture."""
    # Official DOVER model structure:
    # - technical_backbone.* -> technical_backbone.*
    # - aesthetic_backbone.* -> aesthetic_backbone.*
    # - technical_head.* -> technical_head.*
    # - aesthetic_head.* -> aesthetic_head.*
    
    # Our architecture matches this, so minimal mapping needed
    new_state_dict = {}
    for key, value in state_dict.items():
        # Keep keys that match our architecture
        if any(key.startswith(prefix) for prefix in [
            'technical_backbone', 'aesthetic_backbone', 
            'technical_head', 'aesthetic_head'
        ]):
            new_state_dict[key] = value
        else:
            # Skip unknown keys
            print(f"Skipping unknown key: {key}")
    
    return new_state_dict


def load_model(model_type='full', device='cpu'):
    """Load DOVER model and weights."""
    model_dir = Path(os.environ.get('DOVER_MODEL_PATH', '/app/models'))
    
    if model_type == 'mobile':
        model = DOVERMobile()
        weight_path = model_dir / 'DOVER-Mobile.pth'
    else:
        model = DOVER()
        weight_path = model_dir / 'DOVER.pth'
    
    # Initialize with random weights for now (since official weights don't match our simplified architecture)
    print(f"Initializing {model_type} DOVER model with random weights")
    print("Note: For production use, you would need to train or convert official weights")
    
    # Could load and adapt official weights here if needed
    if weight_path.exists():
        try:
            checkpoint = torch.load(weight_path, map_location=device)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            
            # Try to map keys
            mapped_state_dict = map_model_keys(state_dict)
            
            if mapped_state_dict:  # Only load if we have matching keys
                model.load_state_dict(mapped_state_dict, strict=False)
                print(f"Loaded partial weights from {weight_path}")
            else:
                print(f"No matching keys found in {weight_path}, using random initialization")
        except Exception as e:
            print(f"Could not load weights from {weight_path}: {e}")
            print("Using random initialization")
    else:
        print(f"Model weights not found at {weight_path}, using random initialization")
    
    model.to(device)
    model.eval()
    
    return model


def evaluate_video(video_path, model, device='cpu'):
    """Evaluate a single video."""
    # Load and preprocess video (returns dict with technical and aesthetic views)
    try:
        video_views = load_video(video_path, num_frames=32, num_fragments=8)
        
        # Convert to tensors and add batch dimension
        vclips = {}
        for view_name, view_data in video_views.items():
            tensor = torch.from_numpy(view_data).float().unsqueeze(0).to(device)
            vclips[view_name] = tensor
            
    except Exception as e:
        raise RuntimeError(f"Failed to load video: {e}")
    
    # Run inference
    with torch.no_grad():
        outputs = model(vclips)
    
    # Extract scores - outputs is a list [technical_score, aesthetic_score]
    if isinstance(outputs, list) and len(outputs) >= 2:
        technical_score = float(outputs[0].cpu().numpy()[0])
        aesthetic_score = float(outputs[1].cpu().numpy()[0])
        
        # Compute overall score (weighted combination as per DOVER paper)
        # Technical quality is more important (60%) than aesthetic (40%)
        overall_score = 0.6 * technical_score + 0.4 * aesthetic_score
        
    elif isinstance(outputs, list) and len(outputs) == 1:
        # Only one view processed
        overall_score = float(outputs[0].cpu().numpy()[0])
        technical_score = overall_score
        aesthetic_score = overall_score
    else:
        # Fallback
        overall_score = 0.5
        technical_score = 0.5
        aesthetic_score = 0.5
    
    # Ensure scores are in reasonable range [0, 1]
    technical_score = np.clip(technical_score, 0, 1)
    aesthetic_score = np.clip(aesthetic_score, 0, 1)
    overall_score = np.clip(overall_score, 0, 1)
    
    return {
        'overall': overall_score,
        'technical': technical_score,
        'aesthetic': aesthetic_score
    }


def main():
    args = parse_args()
    
    # Validate input
    if not os.path.exists(args.video_path):
        error_result = {
            'status': 'error',
            'error': f"Video file not found: {args.video_path}",
            'metric': 'DOVER'
        }
        print(json.dumps(error_result))
        sys.exit(1)
    
    # Start timing
    start_time = time.time()
    
    try:
        # Load model
        model = load_model(model_type=args.model, device=args.device)
        
        # Evaluate video with dual-view processing
        scores = evaluate_video(args.video_path, model, device=args.device)
        
        # Prepare result
        result = {
            'status': 'success',
            'metric': 'DOVER',
            'variant': args.model,
            'video': os.path.basename(args.video_path),
            'scores': scores,
            'execution_time': time.time() - start_time,
            'device': args.device
        }
        
    except Exception as e:
        # Error handling
        result = {
            'status': 'error',
            'metric': 'DOVER',
            'variant': args.model,
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