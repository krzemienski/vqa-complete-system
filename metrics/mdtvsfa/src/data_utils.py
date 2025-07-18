"""
Data loading utilities for MDTVSFA.
Handles multi-dataset loading and preprocessing for cross-dataset training.
"""

import numpy as np
import torch
import cv2
from typing import List, Tuple, Dict, Optional
import random
from pathlib import Path


# Dataset identifiers
DATASET_IDS = {
    'konvid1k': 0,
    'youtube_ugc': 1,
    'live_vqc': 2,
    'live_qualcomm': 3,
    'cvd2014': 4,
    'generic': 5  # For unknown datasets
}


def get_video_info(video_path: str) -> Tuple[int, float, Tuple[int, int]]:
    """Get video information."""
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return total_frames, fps, (height, width)


def sample_frames_uniform(total_frames: int, num_frames: int) -> List[int]:
    """Sample frames uniformly from video."""
    if total_frames <= num_frames:
        # If video is shorter, use all frames and repeat if needed
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.extend(list(range(total_frames)))
        return indices[:num_frames]
    
    # Uniform sampling
    interval = total_frames / num_frames
    indices = [int(i * interval) for i in range(num_frames)]
    
    return indices


def sample_frames_random(total_frames: int, num_frames: int) -> List[int]:
    """Sample frames randomly from video."""
    if total_frames <= num_frames:
        return sample_frames_uniform(total_frames, num_frames)
    
    # Random sampling without replacement
    indices = sorted(random.sample(range(total_frames), num_frames))
    
    return indices


def load_video_frames(video_path: str, frame_indices: List[int], 
                     target_size: Tuple[int, int] = (224, 224),
                     normalize: bool = True) -> np.ndarray:
    """Load specific frames from video."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize with aspect ratio preservation
            frame = resize_with_padding(frame, target_size)
            
            frames.append(frame)
        else:
            # If frame read fails, use black frame
            frames.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))
    
    cap.release()
    
    frames = np.array(frames, dtype=np.float32)
    
    if normalize:
        # Normalize to [0, 1] and apply ImageNet normalization
        frames = frames / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean) / std
    
    return frames


def resize_with_padding(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image with padding to preserve aspect ratio."""
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    
    # Calculate padding
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    
    # Place resized image
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    
    return padded


def load_video(video_path: str, num_frames: int = 32, 
               sampling: str = 'uniform',
               target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load video with specified sampling strategy.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        sampling: Sampling strategy ('uniform' or 'random')
        target_size: Target frame size
    
    Returns:
        frames: (T, C, H, W) array of frames
    """
    # Get video info
    total_frames, fps, original_size = get_video_info(video_path)
    
    # Sample frame indices
    if sampling == 'uniform':
        indices = sample_frames_uniform(total_frames, num_frames)
    else:
        indices = sample_frames_random(total_frames, num_frames)
    
    # Load frames
    frames = load_video_frames(video_path, indices, target_size)
    
    # Transpose to (T, C, H, W)
    frames = np.transpose(frames, (0, 3, 1, 2))
    
    return frames


def augment_video(frames: np.ndarray, training: bool = True) -> np.ndarray:
    """Apply data augmentation to video frames."""
    if not training:
        return frames
    
    T, C, H, W = frames.shape
    
    # Random horizontal flip
    if random.random() > 0.5:
        frames = frames[:, :, :, ::-1].copy()
    
    # Random brightness adjustment
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        frames = frames * brightness_factor
    
    # Random contrast adjustment
    if random.random() > 0.5:
        contrast_factor = random.uniform(0.8, 1.2)
        mean = frames.mean(axis=(2, 3), keepdims=True)
        frames = (frames - mean) * contrast_factor + mean
    
    # Clip values
    frames = np.clip(frames, -2.5, 2.5)  # Assuming normalized with ImageNet stats
    
    return frames


def get_dataset_id(video_path: str) -> int:
    """Infer dataset ID from video path."""
    path_lower = str(video_path).lower()
    
    for dataset_name, dataset_id in DATASET_IDS.items():
        if dataset_name in path_lower:
            return dataset_id
    
    # Default to generic
    return DATASET_IDS['generic']


class MultiDatasetVideoLoader:
    """Loader for multiple video quality datasets."""
    
    def __init__(self, dataset_paths: Dict[str, str], 
                 num_frames: int = 32,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            dataset_paths: Dict mapping dataset names to paths
            num_frames: Number of frames to sample per video
            target_size: Target frame size
        """
        self.dataset_paths = dataset_paths
        self.num_frames = num_frames
        self.target_size = target_size
        self.videos = []
        self.labels = []
        self.dataset_ids = []
        
        # Load dataset information
        self._load_datasets()
    
    def _load_datasets(self):
        """Load video paths and labels from all datasets."""
        # This would be implemented based on actual dataset formats
        # For now, using placeholder
        pass
    
    def load_video(self, idx: int, training: bool = False) -> Dict[str, torch.Tensor]:
        """Load a video by index."""
        video_path = self.videos[idx]
        label = self.labels[idx]
        dataset_id = self.dataset_ids[idx]
        
        # Load video frames
        frames = load_video(
            video_path, 
            self.num_frames, 
            sampling='random' if training else 'uniform',
            target_size=self.target_size
        )
        
        # Apply augmentation
        frames = augment_video(frames, training)
        
        # Convert to tensor
        frames_tensor = torch.from_numpy(frames).float()
        
        return {
            'frames': frames_tensor,
            'label': torch.tensor(label, dtype=torch.float32),
            'dataset_id': torch.tensor(dataset_id, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        return self.load_video(idx, training=True)


class VideoQualityDataset(torch.utils.data.Dataset):
    """PyTorch dataset for video quality assessment."""
    
    def __init__(self, video_paths: List[str], 
                 labels: Optional[List[float]] = None,
                 num_frames: int = 32,
                 target_size: Tuple[int, int] = (224, 224),
                 transform=None,
                 training: bool = False):
        self.video_paths = video_paths
        self.labels = labels if labels is not None else [0.5] * len(video_paths)
        self.num_frames = num_frames
        self.target_size = target_size
        self.transform = transform
        self.training = training
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        # Load video
        video_path = self.video_paths[idx]
        frames = load_video(
            video_path,
            self.num_frames,
            sampling='random' if self.training else 'uniform',
            target_size=self.target_size
        )
        
        # Apply augmentation
        frames = augment_video(frames, self.training)
        
        # Get dataset ID
        dataset_id = get_dataset_id(video_path)
        
        # Convert to tensors
        frames_tensor = torch.from_numpy(frames).float()
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        dataset_tensor = torch.tensor(dataset_id, dtype=torch.long)
        
        return {
            'frames': frames_tensor,
            'label': label_tensor,
            'dataset_id': dataset_tensor,
            'path': video_path
        }