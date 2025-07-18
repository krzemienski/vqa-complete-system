#!/usr/bin/env python3
"""
Test script for Fast-VQA/FasterVQA implementation.
Tests both variants with synthetic data.
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

from src.fastvqa import FastVQA, FasterVQA
from src.data_utils import load_video_fragments, get_video_info, sample_fragment_indices
import torch


def create_test_video(output_path, duration=2, fps=30, resolution=(640, 480)):
    """Create a synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, resolution)
    
    total_frames = int(duration * fps)
    for i in range(total_frames):
        # Create gradient frame with motion
        frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Moving gradient
        offset = int((i / total_frames) * resolution[0])
        for x in range(resolution[0]):
            color_value = int(255 * ((x + offset) % resolution[0] / resolution[0]))
            frame[:, x, 0] = color_value
            frame[:, x, 1] = 255 - color_value
            frame[:, x, 2] = 128
        
        # Add some noise
        noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        out.write(frame)
    
    out.release()


def test_model_initialization():
    """Test model creation."""
    print("Testing model initialization...")
    
    try:
        # Test FastVQA
        model_fast = FastVQA(num_fragments=8, frames_per_fragment=8)
        print("✓ FastVQA model initialized")
        
        # Test FasterVQA
        model_faster = FasterVQA(num_fragments=4, frames_per_fragment=8)
        print("✓ FasterVQA model initialized")
        
        # Check model structure
        total_params_fast = sum(p.numel() for p in model_fast.parameters())
        total_params_faster = sum(p.numel() for p in model_faster.parameters())
        
        print(f"  FastVQA parameters: {total_params_fast:,}")
        print(f"  FasterVQA parameters: {total_params_faster:,}")
        
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
        
        # Test fragment sampling
        fragment_indices = sample_fragment_indices(total_frames, num_fragments=8, frames_per_fragment=8)
        print(f"✓ Fragment indices sampled: {len(fragment_indices)} fragments")
        
        # Load video fragments
        video_data = load_video_fragments(video_path, num_fragments=8, frames_per_fragment=8)
        print(f"✓ Loaded video data with shape: {video_data.shape}")
        
        # Check shape
        expected_shape = (8, 8, 3, 224, 224)  # (fragments, frames, channels, height, width)
        assert video_data.shape == expected_shape, f"Expected shape {expected_shape}, got {video_data.shape}"
        
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
        # Create models
        model_fast = FastVQA()
        model_faster = FasterVQA()
        
        model_fast.eval()
        model_faster.eval()
        
        # Test FastVQA
        print("\nTesting FastVQA inference...")
        dummy_input_fast = torch.randn(1, 8, 8, 3, 224, 224)  # B, F, T, C, H, W
        
        with torch.no_grad():
            score_fast = model_fast(dummy_input_fast)
        
        print(f"✓ FastVQA inference successful")
        print(f"  Score shape: {score_fast.shape}")
        print(f"  Score value: {score_fast.item():.3f}")
        
        # Test FasterVQA
        print("\nTesting FasterVQA inference...")
        dummy_input_faster = torch.randn(1, 4, 8, 3, 224, 224)  # Fewer fragments
        
        with torch.no_grad():
            score_faster = model_faster(dummy_input_faster)
        
        print(f"✓ FasterVQA inference successful")
        print(f"  Score shape: {score_faster.shape}")
        print(f"  Score value: {score_faster.item():.3f}")
        
        # Check score ranges
        assert 0 <= score_fast.item() <= 1, f"FastVQA score should be in [0, 1], got {score_fast.item()}"
        assert 0 <= score_faster.item() <= 1, f"FasterVQA score should be in [0, 1], got {score_faster.item()}"
        
        print("✓ All scores in valid range [0, 1]")
        
        return True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        return False


def test_batch_processing():
    """Test batch processing capabilities."""
    print("\nTesting batch processing...")
    
    try:
        model = FasterVQA()  # Use faster variant
        model.eval()
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 4, 8, 3, 224, 224)
            
            with torch.no_grad():
                scores = model(dummy_input)
            
            assert scores.shape == (batch_size,), f"Expected shape ({batch_size},), got {scores.shape}"
            print(f"✓ Batch size {batch_size}: output shape {scores.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False


def test_script_execution():
    """Test the main execution script."""
    print("\nTesting script execution...")
    
    try:
        # Create temporary video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            video_path = tmp.name
        
        create_test_video(video_path, duration=1)
        
        # Import run_fastvqa module
        import run_fastvqa
        
        # Temporarily replace sys.argv
        old_argv = sys.argv
        sys.argv = ['run_fastvqa.py', video_path, '--model', 'faster']
        
        # Capture output
        from io import StringIO
        import contextlib
        
        output_buffer = StringIO()
        with contextlib.redirect_stdout(output_buffer):
            try:
                run_fastvqa.main()
            except SystemExit as e:
                if e.code != 0:
                    raise RuntimeError(f"Script exited with code {e.code}")
        
        # Parse output
        output = output_buffer.getvalue()
        result = json.loads(output)
        
        print("✓ Script execution successful")
        print(f"  Result status: {result['status']}")
        
        if result['status'] == 'success':
            print(f"  Score: {result['score']:.3f}")
            print(f"  Execution time: {result['execution_time']:.2f}s")
            print(f"  Speed factor: {result['speed_factor']:.1f}x")
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
    print("Fast-VQA/FasterVQA Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Video Loading", test_video_loading),
        ("Model Inference", test_inference),
        ("Batch Processing", test_batch_processing),
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