# VIDEVAL Research Documentation

## Overview
- **Purpose**: Feature-based VQA tool combining 60 hand-crafted features from multiple metrics
- **Key features**:
  - Integrates BRISQUE, NIQE, VIIDEO, TLVQM features
  - Comprehensive statistical analysis
  - Proven performance on UGC datasets
- **Use cases**: Research benchmarking, comprehensive quality analysis, feature importance studies

## Technical Details
- **Algorithm**: 60-dimensional feature extraction with regression
- **Architecture**:
  - Spatial features: BRISQUE (36), NIQE (12), GM-LOG (10)
  - Temporal features: TLVQM motion statistics
  - VIIDEO naturalness features
  - SVR or Random Forest regression
- **Input requirements**:
  - RGB video
  - Preprocessed to consistent frame rate
  - Works best with 10+ seconds
- **Output format**:
  - Overall quality score (0-100)
  - Individual feature values (optional)
  - Feature importance weights

## Implementation Resources
- **Official repository**: https://github.com/utlive/VIDEVAL_release
- **Papers**:
  - "UGC-VQA: Benchmarking Blind Video Quality Assessment for User Generated Content" (TIP 2021)
- **Documentation**: Comprehensive MATLAB guide
- **Model weights**:
  - Pre-trained regression models
  - Feature normalization parameters

## Implementation Notes
- **Dependencies**:
  - MATLAB R2018b+
  - Image Processing Toolbox
  - Statistics and Machine Learning Toolbox
  - BRISQUE/NIQE implementations
- **Known issues**:
  - High computational cost (60 features)
  - Memory intensive for long videos
  - MATLAB license required
- **Performance optimization**:
  - Parallel feature extraction
  - Frame sampling strategies
  - Pre-computed feature cache
- **Error handling**:
  - Feature extraction failure handling
  - Missing toolbox detection
  - Corrupted frame skipping

## Testing
- **Test cases**:
  - Feature completeness checks (all 60)
  - Cross-validation with paper results
  - Edge cases (short videos, single color)
- **Expected outputs**:
  - Normalized to MOS scale (1-5 or 0-100)
  - Feature vector shape: (60,)
  - Consistent with LIVE-VQC dataset
- **Validation methods**:
  - Feature correlation analysis
  - Ablation studies
  - Dataset-specific calibration

## Code Snippets
```matlab
% Main VIDEVAL pipeline
function [score, features] = compute_videval(video_path)
    % Extract all features
    features = [];
    
    % BRISQUE features (36-dim)
    brisque_feats = extract_brisque_features(video_path);
    features = [features, brisque_feats];
    
    % NIQE features (12-dim)
    niqe_feats = extract_niqe_features(video_path);
    features = [features, niqe_feats];
    
    % GM-LOG features (10-dim)
    gmlog_feats = extract_gmlog_features(video_path);
    features = [features, gmlog_feats];
    
    % TLVQM motion features (2-dim)
    motion_feats = extract_motion_features(video_path);
    features = [features, motion_feats];
    
    % Normalize features
    features_norm = (features - feat_mean) ./ feat_std;
    
    % Predict quality
    score = predict(model, features_norm);
end

% Feature importance
importance = model.feature_importances_;
top_features = get_top_features(importance, feature_names, 10);
```

## References
- IEEE TIP 2021
- LIVE-VQC, KoNViD-1k, YouTube-UGC datasets
- Combines classical and deep features
- Winner of several VQA challenges