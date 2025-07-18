# StableVQA Research Documentation

## Overview
- **Purpose**: Video quality assessment specifically for stabilization artifacts
- **Key features**:
  - Detects shake, jitter, and rolling shutter
  - Analyzes stabilization algorithm effectiveness
  - Includes motion trajectory analysis
- **Use cases**: Mobile video quality, action camera footage, stabilization system evaluation

## Technical Details
- **Algorithm**: Motion analysis with stabilization quality metrics
- **Architecture**:
  - Optical flow estimation
  - Motion trajectory extraction
  - Frequency domain analysis
  - Stabilization artifact detection
  - Quality score regression
- **Input requirements**:
  - Original unstabilized video preferred
  - Works on stabilized video too
  - Higher frame rates better (30+ fps)
- **Output format**:
  - Overall stability score (0-1)
  - Shake magnitude metrics
  - Residual motion analysis
  - Artifact detection flags

## Implementation Resources
- **Official repository**: https://github.com/LIVE-USTC/StableVQA
- **Papers**:
  - "StableVQA: Assessing Video Stabilization Quality" (TMM 2023)
- **Documentation**: Motion analysis guidelines
- **Model weights**:
  - Pre-trained regression model
  - Motion estimation parameters

## Implementation Notes
- **Dependencies**:
  - OpenCV with contrib modules
  - PyTorch for ML components
  - scipy for signal processing
  - Optional: GPU for optical flow
- **Known issues**:
  - Requires good optical flow
  - Can be fooled by intentional motion
  - Frame rate sensitive
- **Performance optimization**:
  - GPU optical flow
  - Pyramid-based motion estimation
  - Cached trajectory computation
- **Error handling**:
  - Handle static videos
  - Optical flow failure detection
  - Extreme motion cases

## Testing
- **Test cases**:
  - Handheld walking videos
  - Drone footage
  - Action camera clips
  - Pre/post stabilization pairs
- **Expected outputs**:
  - Well-stabilized: 0.8-1.0
  - Moderate shake: 0.5-0.8
  - Heavy shake: 0.0-0.5
- **Validation methods**:
  - Synthetic shake injection
  - Known stabilization algorithms
  - User study correlation

## Code Snippets
```python
# StableVQA main pipeline
class StableVQA:
    def __init__(self):
        self.flow_estimator = cv2.optflow.createOptFlow_DeepFlow()
        self.trajectory_analyzer = TrajectoryAnalyzer()
        self.quality_model = load_pretrained_model()
    
    def assess_stability(self, video_path):
        # Extract motion trajectories
        trajectories = self.extract_trajectories(video_path)
        
        # Analyze in frequency domain
        freq_features = self.frequency_analysis(trajectories)
        
        # Detect stabilization artifacts
        artifacts = self.detect_artifacts(trajectories)
        
        # Compute quality features
        features = self.compute_features(
            trajectories, freq_features, artifacts
        )
        
        # Predict quality score
        score = self.quality_model(features)
        
        return {
            'stability_score': score,
            'shake_magnitude': features['shake_mag'],
            'smoothness': features['smoothness'],
            'artifacts': artifacts
        }
    
    def extract_trajectories(self, video_path):
        cap = cv2.VideoCapture(video_path)
        trajectories = []
        prev_gray = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Optical flow
                flow = self.flow_estimator.calc(prev_gray, gray, None)
                
                # Global motion estimation
                motion = self.estimate_global_motion(flow)
                trajectories.append(motion)
            
            prev_gray = gray
        
        return np.array(trajectories)
    
    def frequency_analysis(self, trajectories):
        # FFT for shake frequency
        fft_x = np.fft.fft(trajectories[:, 0])
        fft_y = np.fft.fft(trajectories[:, 1])
        
        # Find dominant frequencies
        freqs = np.fft.fftfreq(len(trajectories))
        
        # Shake typically 1-10 Hz
        shake_band = (freqs > 1) & (freqs < 10)
        shake_energy = np.sum(np.abs(fft_x[shake_band])**2 + 
                             np.abs(fft_y[shake_band])**2)
        
        return {
            'shake_energy': shake_energy,
            'dominant_freq': freqs[np.argmax(np.abs(fft_x))],
            'spectrum': (fft_x, fft_y, freqs)
        }

# Feature computation
def compute_features(trajectories, freq_features, artifacts):
    features = {}
    
    # Trajectory smoothness
    velocity = np.diff(trajectories, axis=0)
    acceleration = np.diff(velocity, axis=0)
    
    features['smoothness'] = 1.0 / (1 + np.std(acceleration))
    features['shake_mag'] = np.std(trajectories)
    
    # Frequency features
    features['shake_freq'] = freq_features['dominant_freq']
    features['shake_energy'] = freq_features['shake_energy']
    
    # Artifact features
    features['rolling_shutter'] = artifacts.get('rolling_shutter', 0)
    features['jello_effect'] = artifacts.get('jello', 0)
    
    return features
```

## References
- IEEE TMM 2023
- Video stabilization literature
- Motion estimation algorithms
- Mobile video quality studies