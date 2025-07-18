#!/bin/bash
#
# Build base Docker images for VQA system
# Usage: ./build_base_images.sh [all|python38|python39|cuda12|matlab]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[BUILD]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for NVIDIA Docker support (for CUDA image)
check_nvidia_docker() {
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        print_warning "NVIDIA Docker runtime not detected. CUDA image may not work properly."
        print_warning "Install nvidia-docker2 for GPU support."
    fi
}

# Build function
build_image() {
    local dockerfile=$1
    local image_name=$2
    local description=$3
    
    print_status "Building $description..."
    
    if docker build -f "$dockerfile" -t "$image_name" .; then
        print_status "✓ Successfully built $image_name"
        
        # Show image size
        size=$(docker images --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}" | grep "$image_name" | awk '{print $2}')
        print_status "  Image size: $size"
    else
        print_error "✗ Failed to build $image_name"
        return 1
    fi
}

# Main build logic
main() {
    local target=${1:-all}
    
    print_status "VQA System - Base Image Builder"
    print_status "Target: $target"
    echo ""
    
    case $target in
        python38)
            build_image "python38.Dockerfile" "vqa-system/python38-base:latest" "Python 3.8 base image"
            ;;
        
        python39)
            build_image "python39.Dockerfile" "vqa-system/python39-base:latest" "Python 3.9 base image"
            ;;
        
        cuda12)
            check_nvidia_docker
            build_image "cuda12.Dockerfile" "vqa-system/cuda12-base:latest" "CUDA 12 base image"
            ;;
        
        matlab)
            print_warning "MATLAB Runtime download may take a long time (~2GB)"
            build_image "matlab.Dockerfile" "vqa-system/matlab-base:latest" "MATLAB Runtime base image"
            ;;
        
        all)
            print_status "Building all base images..."
            echo ""
            
            # Build all images
            build_image "python38.Dockerfile" "vqa-system/python38-base:latest" "Python 3.8 base image"
            echo ""
            
            build_image "python39.Dockerfile" "vqa-system/python39-base:latest" "Python 3.9 base image"
            echo ""
            
            check_nvidia_docker
            build_image "cuda12.Dockerfile" "vqa-system/cuda12-base:latest" "CUDA 12 base image"
            echo ""
            
            print_warning "MATLAB Runtime download may take a long time (~2GB)"
            build_image "matlab.Dockerfile" "vqa-system/matlab-base:latest" "MATLAB Runtime base image"
            ;;
        
        *)
            print_error "Unknown target: $target"
            echo "Usage: $0 [all|python38|python39|cuda12|matlab]"
            exit 1
            ;;
    esac
    
    echo ""
    print_status "Build complete!"
    
    # Show all VQA base images
    echo ""
    print_status "VQA System base images:"
    docker images --filter "reference=vqa-system/*-base" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
}

# Run main function
main "$@"