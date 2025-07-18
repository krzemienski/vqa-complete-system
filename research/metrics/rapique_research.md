# RAPIQUE Research Documentation

## Overview
- **Purpose**: Rapid Hybrid Image and Video Quality Evaluator - combines NSS and deep learning features
- **Key features**:
  - Fast computation using hybrid approach
  - MATLAB-based implementation
  - Works on both images and videos
- **Use cases**: Quick quality screening, hybrid feature extraction, research applications

## Technical Details
- **Algorithm**: Hybrid NSS (Natural Scene Statistics) + CNN features
- **Architecture**:
  - Extracts spatial NSS features frame-wise
  - Temporal pooling across frames
  - SVM regression for quality prediction
  - Pre-trained CNN feature extraction
- **Input requirements**:
  - RGB video input
  - Any resolution (preprocessed internally)
  - Minimum 10 frames recommended
- **Output format**:
  - Single quality score (0-100)
  - Can be normalized to 0-1

## Implementation Resources
- **Official repository**: https://github.com/xiongzhu666/RAPIQUE
- **Papers**:
  - "RAPIQUE: Rapid and Accurate Video Quality Prediction of User Generated Content" (IEEE Open Journal 2021)
- **Documentation**: MATLAB implementation with examples
- **Model weights**:
  - Pre-trained SVM model included
  - CNN features from pre-trained models

## Implementation Notes
- **Dependencies**:
  - MATLAB R2019b or later
  - Computer Vision Toolbox
  - Statistics and Machine Learning Toolbox
  - Pre-trained CNN models
- **Known issues**:
  - MATLAB licensing requirements
  - Memory usage scales with video length
  - Slower than pure deep learning methods
- **Performance optimization**:
  - Frame subsampling for speed
  - Parallel feature extraction
  - Pre-compute NSS features
- **Error handling**:
  - Check MATLAB toolbox availability
  - Handle corrupted frames
  - Validate feature dimensions

## Testing
- **Test cases**:
  - Various resolutions (480p to 4K)
  - Different content types
  - Short clips (< 10 frames) edge case
- **Expected outputs**:
  - High quality content: 70-100
  - Medium quality: 40-70
  - Low quality: 0-40
- **Validation methods**:
  - Compare with VIDEVAL scores
  - Check feature extraction success
  - Verify MATLAB integration

## Code Snippets
```matlab
% Feature extraction
function features = extract_rapique_features(video_path)
    % Read video
    v = VideoReader(video_path);
    
    % Extract NSS features
    nss_features = [];
    while hasFrame(v)
        frame = readFrame(v);
        nss = compute_nss_features(frame);
        nss_features = [nss_features; nss];
    end
    
    % Temporal pooling
    features = [mean(nss_features); std(nss_features)];
    
    % Add CNN features
    cnn_feats = extract_cnn_features(video_path);
    features = [features; cnn_feats];
end

% Quality prediction
score = predict(svm_model, features');
normalized_score = score / 100.0;
```

## References
- IEEE Open Journal of Signal Processing
- Based on BRISQUE for spatial features
- Extends image quality to video domain