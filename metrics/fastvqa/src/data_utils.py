"""
Data loading utilities for Fast-VQA.
Handles video loading with fragment sampling for efficient processing.
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
import cv2
import random
from pathlib import Path


def get_video_info(video_path: str) -> Tuple[int, float, Tuple[int, int]]:
    """Get video information."""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return total_frames, fps, (height, width)


def sample_fragment_indices(total_frames: int, num_fragments: int, 
                          frames_per_fragment: int) -> List[List[int]]:
    """Sample fragment indices uniformly across video."""
    # Calculate fragment stride
    total_sampled_frames = num_fragments * frames_per_fragment
    
    if total_frames <= total_sampled_frames:
        # If video is too short, repeat frames
        indices = list(range(total_frames)) * (total_sampled_frames // total_frames + 1)
        indices = indices[:total_sampled_frames]
    else:
        # Uniform sampling with fragment grouping
        fragment_stride = (total_frames - frames_per_fragment) // max(1, num_fragments - 1)
        indices = []
        
        for i in range(num_fragments):
            start_idx = min(i * fragment_stride, total_frames - frames_per_fragment)
            fragment_indices = list(range(start_idx, start_idx + frames_per_fragment))
            indices.extend(fragment_indices)
    
    # Reshape into fragments
    fragment_indices = []
    for i in range(num_fragments):
        start = i * frames_per_fragment
        end = start + frames_per_fragment
        fragment_indices.append(indices[start:end])
    
    return fragment_indices


def load_frames_from_video(video_path: str, frame_indices: List[int], 
                          target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load specific frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            frames.append(frame)
        else:
            # If frame read fails, use black frame
            frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
    
    cap.release()
    
    return np.array(frames)


def normalize_frames(frames: np.ndarray) -> np.ndarray:
    """Normalize frames to [0, 1] range."""
    frames = frames.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    frames = (frames - mean) / std
    
    return frames


def load_video_fragments(video_path: str, num_fragments: int = 8, 
                        frames_per_fragment: int = 8,
                        target_size: Tuple[int, int] = (224, 224),
                        normalize: bool = True) -> np.ndarray:
    """
    Load video with fragment sampling.
    
    Args:
        video_path: Path to video file
        num_fragments: Number of fragments to sample
        frames_per_fragment: Number of frames per fragment
        target_size: Target frame size (H, W)
        normalize: Whether to normalize frames
    
    Returns:
        video_data: (F, T, C, H, W) - Fragments, Time, Channels, Height, Width
    """
    # Get video info
    total_frames, fps, original_size = get_video_info(video_path)
    
    # Sample fragment indices
    fragment_indices = sample_fragment_indices(total_frames, num_fragments, frames_per_fragment)
    
    # Load fragments
    fragments = []
    for indices in fragment_indices:
        frames = load_frames_from_video(video_path, indices, target_size)
        
        if normalize:
            frames = normalize_frames(frames)
        
        # Transpose to (T, C, H, W)
        frames = np.transpose(frames, (0, 3, 1, 2))
        fragments.append(frames)
    
    # Stack fragments: (F, T, C, H, W)
    video_data = np.stack(fragments, axis=0)
    
    return video_data


def augment_video_fragments(fragments: np.ndarray, training: bool = False) -> np.ndarray:
    """Apply data augmentation to video fragments."""
    if not training:
        return fragments
    
    F, T, C, H, W = fragments.shape
    
    # Random horizontal flip
    if random.random() > 0.5:
        fragments = fragments[:, :, :, :, ::-1].copy()
    
    # Random crop (simulate by slight zoom)
    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.0)
        new_H = int(H * scale)
        new_W = int(W * scale)
        
        # Center crop
        start_h = (H - new_H) // 2
        start_w = (W - new_W) // 2
        
        cropped = fragments[:, :, :, start_h:start_h+new_H, start_w:start_w+new_W]
        
        # Resize back
        resized_fragments = []
        for f in range(F):
            fragment_frames = []
            for t in range(T):
                frame = cropped[f, t].transpose(1, 2, 0)  # C, H, W -> H, W, C
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
                frame = frame.transpose(2, 0, 1)  # H, W, C -> C, H, W
                fragment_frames.append(frame)
            resized_fragments.append(np.stack(fragment_frames))
        
        fragments = np.stack(resized_fragments)
    
    return fragments


def create_temporal_mask(num_frames: int, mask_ratio: float = 0.0) -> torch.Tensor:
    """Create temporal mask for masked frame modeling."""
    if mask_ratio <= 0:
        return torch.ones(num_frames, dtype=torch.bool)
    
    num_masked = int(num_frames * mask_ratio)
    mask = torch.ones(num_frames, dtype=torch.bool)
    
    # Random masking
    masked_indices = torch.randperm(num_frames)[:num_masked]
    mask[masked_indices] = False
    
    return mask


class VideoFragmentDataset(torch.utils.data.Dataset):
    """PyTorch dataset for video fragments."""
    
    def __init__(self, video_paths: List[str], labels: Optional[List[float]] = None,
                 num_fragments: int = 8, frames_per_fragment: int = 8,
                 transform=None, training: bool = False):
        self.video_paths = video_paths
        self.labels = labels if labels is not None else [0.5] * len(video_paths)
        self.num_fragments = num_fragments
        self.frames_per_fragment = frames_per_fragment
        self.transform = transform
        self.training = training
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        # Load video fragments
        video_path = self.video_paths[idx]
        fragments = load_video_fragments(
            video_path, 
            self.num_fragments, 
            self.frames_per_fragment
        )
        
        # Apply augmentation
        fragments = augment_video_fragments(fragments, self.training)
        
        # Convert to tensor
        fragments_tensor = torch.from_numpy(fragments).float()
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return fragments_tensor, label