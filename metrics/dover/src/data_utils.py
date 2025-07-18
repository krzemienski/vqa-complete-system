"""
Data utilities for DOVER video loading and preprocessing.
"""

import numpy as np
import torch
import cv2
from pathlib import Path

# Try to import decord, fall back to cv2 if not available
try:
    from decord import VideoReader, cpu
    from decord.bridge import to_torch
    HAS_DECORD = True
except ImportError:
    print("Warning: decord not available, using cv2 for video loading")
    HAS_DECORD = False


class VideoDataset(torch.utils.data.Dataset):
    """Dataset for loading videos with dual-view processing."""
    
    def __init__(self, video_paths, num_frames=32, num_fragments=8, size=(224, 224)):
        self.video_paths = video_paths if isinstance(video_paths, list) else [video_paths]
        self.num_frames = num_frames
        self.num_fragments = num_fragments
        self.size = size
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_views = load_video(
            video_path,
            num_frames=self.num_frames,
            num_fragments=self.num_fragments,
            size=self.size
        )
        
        # Convert each view to tensor
        tensor_views = {}
        for view_name, view_data in video_views.items():
            tensor_views[view_name] = torch.from_numpy(view_data).float()
            
        return tensor_views


def load_video(video_path, num_frames=32, num_fragments=8, size=(224, 224)):
    """
    Load video and extract both technical and aesthetic views for DOVER.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames per fragment
        num_fragments: Number of fragments to extract
        size: Target size (H, W) for aesthetic view
    
    Returns:
        dict with 'technical' and 'aesthetic' views:
        - technical: (fragments, frames, channels, height, width) - fragment view
        - aesthetic: (frames, channels, height, width) - resized view
    """
    if HAS_DECORD:
        return _load_video_dual_view_decord(video_path, num_frames, num_fragments, size)
    else:
        return _load_video_dual_view_cv2(video_path, num_frames, num_fragments, size)


def _load_video_dual_view_decord(video_path, num_frames, num_fragments, aesthetic_size):
    """Load video with both technical and aesthetic views using decord."""
    # Open video
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)
    
    # === TECHNICAL VIEW (fragments) ===
    # Fragment-based sampling for technical quality
    technical_size = (32, 32)  # Small fragments as per config
    
    # Calculate fragment positions
    if total_frames < num_frames * num_fragments:
        fragment_stride = max(1, total_frames // num_fragments)
        fragment_starts = [i * fragment_stride for i in range(num_fragments)]
    else:
        available_frames = total_frames - num_frames
        fragment_stride = available_frames // (num_fragments - 1) if num_fragments > 1 else 0
        fragment_starts = [i * fragment_stride for i in range(num_fragments)]
    
    # Extract technical fragments
    technical_fragments = []
    
    for start_idx in fragment_starts:
        end_idx = min(start_idx + num_frames, total_frames)
        
        if end_idx - start_idx < num_frames:
            indices = list(range(start_idx, end_idx))
            indices += [end_idx - 1] * (num_frames - len(indices))
        else:
            indices = list(range(start_idx, end_idx))[:num_frames]
        
        # Load and resize for technical view
        frames = vr.get_batch(indices).asnumpy()
        resized_frames = []
        for frame in frames:
            frame_resized = cv2.resize(frame, technical_size, interpolation=cv2.INTER_LINEAR)
            resized_frames.append(frame_resized)
        
        frames = np.stack(resized_frames)
        frames = frames.astype(np.float32) / 255.0
        frames = frames.transpose(0, 3, 1, 2)  # (T, C, H, W)
        technical_fragments.append(frames)
    
    technical_view = np.stack(technical_fragments)  # (F, T, C, H, W)
    
    # === AESTHETIC VIEW (resized) ===
    # Uniform sampling for aesthetic quality
    if total_frames >= num_frames:
        # Sample uniformly across the video
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Repeat frames if video is too short
        indices = np.arange(total_frames)
        indices = np.tile(indices, (num_frames // total_frames) + 1)[:num_frames]
    
    # Load and resize for aesthetic view
    aesthetic_frames = vr.get_batch(indices.tolist()).asnumpy()
    resized_aesthetic = []
    for frame in aesthetic_frames:
        frame_resized = cv2.resize(frame, aesthetic_size, interpolation=cv2.INTER_LINEAR)
        resized_aesthetic.append(frame_resized)
    
    aesthetic_frames = np.stack(resized_aesthetic)
    aesthetic_frames = aesthetic_frames.astype(np.float32) / 255.0
    aesthetic_view = aesthetic_frames.transpose(0, 3, 1, 2)  # (T, C, H, W)
    
    return {
        'technical': technical_view,   # (F, T, C, H, W)
        'aesthetic': aesthetic_view    # (T, C, H, W)
    }


def _load_video_dual_view_cv2(video_path, num_frames, num_fragments, aesthetic_size):
    """Load video with both technical and aesthetic views using cv2."""
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        raise ValueError(f"Video file has no frames: {video_path}")
    
    # === TECHNICAL VIEW (fragments) ===
    technical_size = (32, 32)  # Small fragments
    
    # Calculate fragment positions
    if total_frames < num_frames * num_fragments:
        fragment_stride = max(1, total_frames // num_fragments)
        fragment_starts = [i * fragment_stride for i in range(num_fragments)]
    else:
        available_frames = total_frames - num_frames
        fragment_stride = available_frames // (num_fragments - 1) if num_fragments > 1 else 0
        fragment_starts = [i * fragment_stride for i in range(num_fragments)]
    
    # Extract technical fragments
    technical_fragments = []
    
    for start_idx in fragment_starts:
        end_idx = min(start_idx + num_frames, total_frames)
        
        if end_idx - start_idx < num_frames:
            indices = list(range(start_idx, end_idx))
            indices += [end_idx - 1] * (num_frames - len(indices))
        else:
            indices = list(range(start_idx, end_idx))[:num_frames]
        
        # Load frames for technical view
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                if frames:
                    frame = frames[-1].copy()
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Convert BGR to RGB and resize for technical view
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame, technical_size, interpolation=cv2.INTER_LINEAR)
            frames.append(frame_resized)
        
        frames = np.stack(frames)
        frames = frames.astype(np.float32) / 255.0
        frames = frames.transpose(0, 3, 1, 2)  # (T, C, H, W)
        technical_fragments.append(frames)
    
    technical_view = np.stack(technical_fragments)  # (F, T, C, H, W)
    
    # === AESTHETIC VIEW (resized) ===
    # Uniform sampling for aesthetic quality
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.arange(total_frames)
        indices = np.tile(indices, (num_frames // total_frames) + 1)[:num_frames]
    
    # Load frames for aesthetic view
    aesthetic_frames = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            if aesthetic_frames:
                frame = aesthetic_frames[-1].copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Convert BGR to RGB and resize for aesthetic view
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame, aesthetic_size, interpolation=cv2.INTER_LINEAR)
        aesthetic_frames.append(frame_resized)
    
    aesthetic_frames = np.stack(aesthetic_frames)
    aesthetic_frames = aesthetic_frames.astype(np.float32) / 255.0
    aesthetic_view = aesthetic_frames.transpose(0, 3, 1, 2)  # (T, C, H, W)
    
    # Release video capture
    cap.release()
    
    return {
        'technical': technical_view,   # (F, T, C, H, W)
        'aesthetic': aesthetic_view    # (T, C, H, W)
    }


def preprocess_video(video_tensor):
    """
    Preprocess video tensor for DOVER.
    
    Args:
        video_tensor: Tensor of shape (F, T, C, H, W) or (T, C, H, W)
    
    Returns:
        Normalized tensor
    """
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    
    if video_tensor.dim() == 5:
        # Has fragments dimension
        mean = mean.view(1, 1, 3, 1, 1)
        std = std.view(1, 1, 3, 1, 1)
    else:
        # No fragments
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    
    video_tensor = (video_tensor - mean) / std
    
    return video_tensor