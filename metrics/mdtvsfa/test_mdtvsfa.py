#!/usr/bin/env python3
"""
Test script for MDTVSFA implementation.
Tests both full and lite variants with synthetic data.
"""

import os
import sys
import json
import tempfile
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.mdtvsfa import MDTVSFA, MDTVSFALite
from src.data_utils import load_video, get_video_info, DATASET_IDS
import torch


def create_test_video(output_path, duration=2, fps=30, resolution=(640, 480)):
    """Create a synthetic test video with varied content."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    
    total_frames = int(duration * fps)
    for i in range(total_frames):
        # Create frame with motion and spatial patterns
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Moving patterns
        t = i / total_frames
        
        # Horizontal gradient
        for x in range(resolution[0]):
            frame[:, x, 0] = int(255 * (x / resolution[0]))
        
        # Vertical gradient
        for y in range(resolution[1]):
            frame[y, :, 1] = int(255 * (y / resolution[1]))
        
        # Moving circle
        cx = int(resolution[0] * (0.5 + 0.3 * np.sin(2 * np.pi * t)))
        cy = int(resolution[1] * (0.5 + 0.3 * np.cos(2 * np.pi * t)))
        cv2.circle(frame, (cx, cy), 50, (255, 255, 255), -1)
        
        # Add noise
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()


def test_model_initialization():
    """Test model creation."""
    print("Testing model initialization...")
    
    try:
        # Test full model
        model_full = MDTVSFA(num_datasets=6)
        print("✓ MDTVSFA full model initialized")
        
        # Test lite model
        model_lite = MDTVSFALite(num_datasets=6)
        print("✓ MDTVSFA lite model initialized")
        
        # Check model parameters
        total_params_full = sum(p.numel() for p in model_full.parameters())
        total_params_lite = sum(p.numel() for p in model_lite.parameters())
        
        print(f"  Full model parameters: {total_params_full:,}")
        print(f"  Lite model parameters: {total_params_lite:,}")
        
        # Check dataset calibration layers
        assert len(model_full.dataset_calibration) == 6
        print(f"  Dataset calibration layers: {len(model_full.dataset_calibration)}")
        
        return True
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False


def test_video_loading():
    """Test video loading functionality."""
    print("\nTesting video loading...")
    
    try:
        # Create temporary video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        create_test_video(video_path)
        print(f"✓ Created test video at {video_path}")
        
        # Test video info
        total_frames, fps, size = get_video_info(video_path)
        print(f"✓ Video info: {total_frames} frames @ {fps} fps, size: {size}")
        
        # Load video
        video_data = load_video(video_path, num_frames=32)
        print(f"✓ Loaded video data with shape: {video_data.shape}")
        
        # Check shape
        expected_shape = (32, 3, 224, 224)  # (T, C, H, W)
        assert video_data.shape == expected_shape, f"Expected shape {expected_shape}, got {video_data.shape}"
        
        # Test dataset ID inference
        dataset_id = get_dataset_id(video_path)
        print(f"✓ Dataset ID inferred: {dataset_id} (generic)")
        
        # Clean up
        os.unlink(video_path)
        
        return True
    except Exception as e:
        print(f"✗ Video loading failed: {e}")
        if 'video_path' in locals() and os.path.exists(video_path):
            os.unlink(video_path)
        return False


def test_feature_extraction():
    """Test feature extraction components."""
    print("\nTesting feature extraction...")
    
    try:
        model = MDTVSFA()
        model.eval()
        
        # Create dummy input
        batch_size = 2
        num_frames = 16  # Smaller for testing
        dummy_input = torch.randn(batch_size, num_frames, 3, 224, 224)
        
        # Extract features
        spatial_features, motion_features = model.extract_features(dummy_input)
        
        print(f"✓ Feature extraction successful")
        print(f"  Spatial features shape: {spatial_features.shape}")
        print(f"  Motion features shape: {motion_features.shape}")
        
        # Check shapes
        assert spatial_features.shape == (batch_size, num_frames, 2048)
        assert motion_features.shape == (batch_size, num_frames, 128)
        
        return True
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        return False


def test_inference():
    """Test model inference."""
    print("\nTesting model inference...")
    
    try:
        # Test full model
        print("\nTesting MDTVSFA full model...")
        model_full = MDTVSFA()
        model_full.eval()
        
        dummy_input = torch.randn(1, 32, 3, 224, 224)
        
        # Test without dataset ID
        with torch.no_grad():
            score = model_full(dummy_input)
        
        print(f"✓ Full model inference (no dataset ID)")
        print(f"  Score: {score.item():.3f}")
        
        # Test with dataset ID and features
        with torch.no_grad():
            score, features = model_full(dummy_input, dataset_id=0, return_features=True)
        
        print(f"✓ Full model inference (with dataset ID and features)")
        print(f"  Score: {score.item():.3f}")
        print(f"  Feature keys: {list(features.keys())}")
        
        # Test lite model
        print("\nTesting MDTVSFA lite model...")
        model_lite = MDTVSFALite()
        model_lite.eval()
        
        with torch.no_grad():
            score_lite = model_lite(dummy_input)
        
        print(f"✓ Lite model inference")
        print(f"  Score: {score_lite.item():.3f}")
        
        # Check score ranges
        assert 0 <= score.item() <= 1, f"Score should be in [0, 1], got {score.item()}"
        assert 0 <= score_lite.item() <= 1, f"Score should be in [0, 1], got {score_lite.item()}"
        
        return True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing capabilities."""
    print("\nTesting batch processing...")
    
    try:
        model = MDTVSFALite()  # Use lite for speed
        model.eval()
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 32, 3, 224, 224)
            
            with torch.no_grad():
                scores = model(dummy_input)
            
            assert scores.shape == (batch_size,), f"Expected shape ({batch_size},), got {scores.shape}"
            print(f"✓ Batch size {batch_size}: output shape {scores.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False


def test_dataset_calibration():
    """Test dataset-specific calibration."""
    print("\nTesting dataset calibration...")
    
    try:
        model = MDTVSFA()
        model.eval()
        
        dummy_input = torch.randn(1, 32, 3, 224, 224)
        
        # Test different dataset IDs
        scores = {}
        with torch.no_grad():
            for dataset_name, dataset_id in DATASET_IDS.items():
                if dataset_id < len(model.dataset_calibration):
                    score = model(dummy_input, dataset_id=dataset_id)
                    scores[dataset_name] = score.item()
        
        print("✓ Dataset calibration test successful")
        for dataset_name, score in scores.items():
            print(f"  {dataset_name}: {score:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset calibration test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MDTVSFA Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Video Loading", test_video_loading),
        ("Feature Extraction", test_feature_extraction),
        ("Model Inference", test_inference),
        ("Batch Processing", test_batch_processing),
        ("Dataset Calibration", test_dataset_calibration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    return passed_tests == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)