#!/usr/bin/env python3
"""
Test script for DOVER implementation.
Tests both full and mobile variants with synthetic data.
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

from src.dover import DOVER, DOVERMobile
from src.data_utils import load_video, preprocess_video
import torch


def create_test_video(output_path, duration=2, fps=30, resolution=(640, 480)):
    """Create a synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    
    total_frames = int(duration * fps)
    for i in range(total_frames):
        # Create gradient frame
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Add some variation
        color_value = int(255 * (i / total_frames))
        frame[:, :, 0] = color_value  # Blue channel gradient
        frame[:, :, 1] = 255 - color_value  # Green channel inverse gradient
        frame[:, :, 2] = 128  # Red channel constant
        
        # Add some noise
        noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()


def test_model_initialization():
    """Test model creation."""
    print("Testing model initialization...")
    
    try:
        # Test full model
        model_full = DOVER()
        print("✓ DOVER full model initialized")
        
        # Test mobile model
        model_mobile = DOVERMobile()
        print("✓ DOVER mobile model initialized")
        
        # Check model structure
        total_params_full = sum(p.numel() for p in model_full.parameters())
        total_params_mobile = sum(p.numel() for p in model_mobile.parameters())
        
        print(f"  Full model parameters: {total_params_full:,}")
        print(f"  Mobile model parameters: {total_params_mobile:,}")
        
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
        
        # Load video
        video_data = load_video(video_path, num_frames=32, num_fragments=8)
        print(f"✓ Loaded video data with shape: {video_data.shape}")
        
        # Check shape
        expected_shape = (8, 32, 3, 224, 224)  # (fragments, frames, channels, height, width)
        assert video_data.shape == expected_shape, f"Expected shape {expected_shape}, got {video_data.shape}"
        print(f"✓ Video shape correct: {video_data.shape}")
        
        # Check value range
        assert video_data.min() >= 0 and video_data.max() <= 1, "Video data should be normalized to [0, 1]"
        print(f"✓ Video value range: [{video_data.min():.3f}, {video_data.max():.3f}]")
        
        # Clean up
        os.unlink(video_path)
        
        return True
    except Exception as e:
        print(f"✗ Video loading failed: {e}")
        if 'video_path' in locals() and os.path.exists(video_path):
            os.unlink(video_path)
        return False


def test_inference():
    """Test model inference."""
    print("\nTesting model inference...")
    
    try:
        # Create model
        model = DOVERMobile()  # Use mobile for faster testing
        model.eval()
        
        # Create dummy input
        batch_size = 1
        fragments = 8
        frames = 32
        channels = 3
        height = 224
        width = 224
        
        dummy_input = torch.randn(batch_size, fragments, frames, channels, height, width)
        print(f"✓ Created dummy input with shape: {dummy_input.shape}")
        
        # Run inference
        with torch.no_grad():
            outputs = model(dummy_input)
        
        print("✓ Model inference successful")
        
        # Check outputs
        assert isinstance(outputs, dict), "Output should be a dictionary"
        assert 'technical' in outputs, "Output should contain 'technical' score"
        assert 'aesthetic' in outputs, "Output should contain 'aesthetic' score"
        assert 'overall' in outputs, "Output should contain 'overall' score"
        
        print(f"  Technical score: {outputs['technical'].item():.3f}")
        print(f"  Aesthetic score: {outputs['aesthetic'].item():.3f}")
        print(f"  Overall score: {outputs['overall'].item():.3f}")
        
        # Check score ranges
        for key, score in outputs.items():
            assert 0 <= score.item() <= 1, f"{key} score should be in [0, 1], got {score.item()}"
        
        print("✓ All scores in valid range [0, 1]")
        
        return True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False


def test_script_execution():
    """Test the main execution script."""
    print("\nTesting script execution...")
    
    try:
        # Create temporary video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        create_test_video(video_path, duration=1)
        
        # Import run_dover module
        import run_dover
        
        # Mock command line arguments
        class Args:
            video_path = video_path
            model = 'mobile'
            device = 'cpu'
            output = None
            batch_size = 1
            num_workers = 0
        
        # Temporarily replace sys.argv
        old_argv = sys.argv
        sys.argv = ['run_dover.py', video_path, '--model', 'mobile']
        
        # Capture output
        from io import StringIO
        import contextlib
        
        output_buffer = StringIO()
        with contextlib.redirect_stdout(output_buffer):
            try:
                run_dover.main()
            except SystemExit as e:
                if e.code != 0:
                    raise RuntimeError(f"Script exited with code {e.code}")
        
        # Parse output
        output = output_buffer.getvalue()
        result = json.loads(output)
        
        print("✓ Script execution successful")
        print(f"  Result status: {result['status']}")
        
        if result['status'] == 'success':
            print(f"  Overall score: {result['scores']['overall']:.3f}")
            print(f"  Technical score: {result['scores']['technical']:.3f}")
            print(f"  Aesthetic score: {result['scores']['aesthetic']:.3f}")
        else:
            print(f"  Error: {result.get('error', 'Unknown error')}")
        
        # Restore sys.argv
        sys.argv = old_argv
        
        # Clean up
        os.unlink(video_path)
        
        return result['status'] == 'success'
        
    except Exception as e:
        print(f"✗ Script execution test failed: {e}")
        if 'video_path' in locals() and os.path.exists(video_path):
            os.unlink(video_path)
        if 'old_argv' in locals():
            sys.argv = old_argv
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DOVER Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Video Loading", test_video_loading),
        ("Model Inference", test_inference),
        # Note: Script execution test requires model weights
        # ("Script Execution", test_script_execution),
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