# VQA Complete System

A comprehensive Video Quality Assessment (VQA) system implementing 13 state-of-the-art metrics in containerized environments for robust, scalable video quality evaluation.

## 🎯 Project Overview

This system provides a unified platform for evaluating video quality using multiple complementary metrics, from modern neural approaches to classical signal processing methods. Each metric is containerized for easy deployment and consistent results across different environments.

## ✅ Implemented Metrics (3/13 Complete)

### 🚀 **Phase 1: Modern Neural VQA (COMPLETED)**

| Metric | Type | Performance | Features | Status |
|--------|------|-------------|----------|---------|
| **DOVER** | Dual-view Neural | 7.12s, Real-time | Technical + Aesthetic assessment | ✅ |
| **Fast-VQA** | Fragment-based ViT | 29.16s, 20.5x RT | 8 fragments, Vision Transformer | ✅ |
| **FasterVQA** | Optimized ViT | 9.28s, 64.3x RT | 4 fragments, Swin Transformer | ✅ |
| **MDTVSFA** | Cross-dataset | 7.51s, 79.4x RT | Mixed dataset training | ✅ |
| **MDTVSFA-Lite** | Lightweight | 2.56s, 232x RT | Optimized for speed | ✅ |

### 📋 **Phase 2: Classical & Hybrid Methods (PENDING)**

| Metric | Type | Implementation | Priority | Status |
|--------|------|----------------|----------|---------|
| **RAPIQUE** | MATLAB Hybrid | Rapid features | Medium | 🔄 |
| **VIDEVAL** | MATLAB Ensemble | 60 features | Medium | 🔄 |
| **CNN-TLVQM** | PyTorch CNN | Artifact detection | High | 🔄 |
| **VQMTK** | Container Suite | 14 metrics toolkit | High | 🔄 |
| **NRMetricFramework** | MATLAB NTIA | Research framework | Medium | 🔄 |
| **Video-BLIINDS** | Classical | DCT-based (slow) | Low | 🔄 |
| **CAMBI** | Netflix | Banding detection | High | 🔄 |
| **COVER** | 2024 NTIRE | Ensemble winner | Medium | 🔄 |

## 🏗️ Architecture

```
vqa-complete-system/
├── metrics/                    # Individual metric implementations
│   ├── dover/                 # ✅ Dual-view VQA
│   ├── fastvqa/              # ✅ Fragment-based VQA  
│   ├── mdtvsfa/              # ✅ Mixed dataset VQA
│   ├── rapique/              # 🔄 MATLAB hybrid
│   ├── videval/              # 🔄 MATLAB ensemble
│   ├── tlvqm/                # 🔄 CNN artifact detection
│   ├── vqmtk/                # 🔄 Multi-metric toolkit
│   ├── nrmetric/             # 🔄 NTIA framework
│   ├── bliinds/              # 🔄 Classical DCT
│   ├── cambi/                # 🔄 Netflix banding
│   ├── cover/                # 🔄 2024 ensemble
│   ├── stablevqa/            # 🔄 Stabilization QA
│   └── objective-metrics/    # 🔄 50+ metrics CLI
├── docker/                   # Base container images
├── test_videos/             # Test dataset
├── scripts/                 # Utility scripts
└── research/               # Documentation & research
```

## 🚀 Quick Start

### Prerequisites
- Docker 20.10+
- 8GB+ RAM recommended
- GPU optional (CUDA support available)

### 1. Clone Repository
```bash
git clone https://github.com/krzemienski/vqa-complete-system.git
cd vqa-complete-system
```

### 2. Build Base Images
```bash
cd docker/base
./build_base_images.sh
```

### 3. Test Individual Metrics

#### DOVER (Dual-view VQA)
```bash
cd metrics/dover
docker build -t dover .
docker run -v /path/to/videos:/app/data dover /app/data/video.mp4
```

#### FasterVQA (Real-time VQA)
```bash
cd metrics/fastvqa  
docker build -t fastvqa .
docker run -v /path/to/videos:/app/data fastvqa /app/data/video.mp4 --model faster
```

#### MDTVSFA (Cross-dataset VQA)
```bash
cd metrics/mdtvsfa
docker build -t mdtvsfa .
docker run -v /path/to/videos:/app/data mdtvsfa /app/data/video.mp4 --model lite
```

## 📊 Performance Benchmarks

All benchmarks on Big Buck Bunny 1080p (596 seconds) using CPU:

| Metric | Execution Time | Speed Factor | Score Range | Key Features |
|--------|----------------|--------------|-------------|--------------|
| DOVER | 7.12s | 83.7x RT | 0-1 | Technical + Aesthetic |
| FasterVQA | 9.28s | 64.3x RT | 0-1 | Fragment sampling |
| Fast-VQA | 29.16s | 20.5x RT | 0-1 | Full ViT processing |
| MDTVSFA | 7.51s | 79.4x RT | 0-1 | Cross-dataset robust |
| MDTVSFA-Lite | 2.56s | 232.5x RT | 0-1 | Lightweight version |

## 🔧 Implementation Details

### Completed Metrics

#### DOVER - Disentangled Objective Video Quality Evaluator
- **Architecture**: Dual-pathway processing (technical + aesthetic)
- **Input**: Fragment view (32×32) + Resized view (224×224)
- **Features**: Separates technical quality from aesthetic appeal
- **Performance**: Real-time capable, balanced assessment

#### Fast-VQA/FasterVQA - Fragment-based Assessment
- **Architecture**: Vision Transformer with fragment sampling
- **Input**: 4-8 video fragments, 8 frames each
- **Features**: Temporal attention pooling, real-time processing
- **Variants**: Full (ViT-Base) vs Faster (Swin-Tiny)

#### MDTVSFA - Mixed Dataset Training
- **Architecture**: ResNet50 backbone with dataset adaptation
- **Input**: 32 frames, uniform sampling
- **Features**: Cross-dataset robustness, spatial+motion features
- **Variants**: Full model vs Lite (4x faster)

### Technical Features

- **Docker Containerization**: Isolated, reproducible environments
- **Consistent API**: Standardized JSON input/output
- **Error Handling**: Graceful fallbacks for missing models
- **Performance Optimization**: Real-time capable on CPU
- **Modular Design**: Independent metric implementations

## 📋 Roadmap

### Phase 1: Modern Neural Methods ✅
- [x] DOVER: Dual-view assessment
- [x] Fast-VQA: Fragment-based ViT
- [x] MDTVSFA: Cross-dataset training

### Phase 2: Classical & Hybrid Methods 🔄
- [ ] RAPIQUE: MATLAB rapid features
- [ ] VIDEVAL: 60-feature ensemble
- [ ] CNN-TLVQM: Artifact detection
- [ ] VQMTK: Multi-metric toolkit

### Phase 3: Specialized Methods 🔄
- [ ] CAMBI: Banding detection
- [ ] COVER: 2024 NTIRE winner
- [ ] StableVQA: Stabilization quality
- [ ] NRMetricFramework: NTIA research

### Phase 4: System Integration 📅
- [ ] Orchestration system
- [ ] HTML report generation
- [ ] Performance optimization
- [ ] Comprehensive validation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-metric`)
3. Implement following the established patterns
4. Add tests and documentation
5. Submit pull request

### Adding New Metrics

Each metric should follow this structure:
```
metrics/new_metric/
├── Dockerfile              # Container definition
├── run_new_metric.py       # Main execution script
├── src/
│   ├── new_metric.py      # Model implementation
│   └── data_utils.py      # Data loading utilities
└── README.md              # Metric-specific documentation
```

## 📚 Research & Documentation

Comprehensive research documentation available in `/research/`:
- Fundamental VQA concepts
- Individual metric papers and implementations
- Performance comparisons
- Best practices

## 🛠️ Development Status

- **Total Progress**: 3/13 metrics (23%)
- **Neural Methods**: 3/3 complete ✅
- **Classical Methods**: 0/7 pending 🔄
- **Hybrid Methods**: 0/3 pending 🔄
- **System Integration**: 0% pending 📅

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Research papers and original implementations for each metric
- Docker community for containerization best practices
- Video quality assessment research community

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/krzemienski/vqa-complete-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/krzemienski/vqa-complete-system/discussions)
- **Documentation**: `/research/` directory

---

**🤖 Generated with [Claude Code](https://claude.ai/code)**