#!/usr/bin/env python3
"""
Check the implementation progress of the VQA system.
"""

import os
import json
from pathlib import Path
import subprocess
from datetime import datetime


def check_file_exists(filepath):
    """Check if a file exists and return its size."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        return True, size
    return False, 0


def check_docker_image(image_name):
    """Check if a Docker image exists."""
    try:
        result = subprocess.run(
            ['docker', 'images', '--format', '{{.Repository}}:{{.Tag}}'],
            capture_output=True, text=True, check=True
        )
        return image_name in result.stdout
    except:
        return False


def check_metric_implementation(metric_name, metric_dir):
    """Check the implementation status of a metric."""
    status = {
        'name': metric_name,
        'directory': str(metric_dir),
        'files': {},
        'docker': False,
        'status': 'not_started'
    }
    
    # Check key files
    key_files = {
        'src/model': metric_dir / 'src' / f'{metric_name.lower()}.py',
        'src/data_utils': metric_dir / 'src' / 'data_utils.py',
        'runner': metric_dir / f'run_{metric_name.lower()}.py',
        'dockerfile': metric_dir / 'Dockerfile',
        'test': metric_dir / f'test_{metric_name.lower()}.py',
        'download_script': metric_dir / 'download_models.sh'
    }
    
    files_exist = 0
    for file_type, filepath in key_files.items():
        exists, size = check_file_exists(filepath)
        status['files'][file_type] = {
            'exists': exists,
            'size': size,
            'path': str(filepath)
        }
        if exists:
            files_exist += 1
    
    # Determine implementation status
    if files_exist == 0:
        status['status'] = 'not_started'
    elif files_exist < len(key_files):
        status['status'] = 'in_progress'
    else:
        status['status'] = 'implemented'
    
    # Check Docker image
    docker_image = f'vqa-system/{metric_name.lower()}:latest'
    status['docker'] = check_docker_image(docker_image)
    
    return status


def main():
    """Check overall system progress."""
    base_dir = Path(__file__).parent
    
    # Define all metrics
    metrics = [
        ('DOVER', 'dover'),
        ('Fast-VQA', 'fastvqa'),
        ('MDTVSFA', 'mdtvsfa'),
        ('RAPIQUE', 'rapique'),
        ('VIDEVAL', 'videval'),
        ('CNN-TLVQM', 'tlvqm'),
        ('VQMTK', 'vqmtk'),
        ('NRMetricFramework', 'nrmetric'),
        ('Video-BLIINDS', 'bliinds'),
        ('CAMBI', 'cambi'),
        ('COVER', 'cover'),
        ('StableVQA', 'stablevqa'),
        ('Objective-Metrics', 'objective_metrics')
    ]
    
    # Check base components
    print("=" * 80)
    print("VQA System Implementation Progress Check")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check base Docker images
    print("\n📦 Base Docker Images:")
    base_images = [
        'vqa-system/python38-base:latest',
        'vqa-system/python39-base:latest',
        'vqa-system/cuda12-base:latest',
        'vqa-system/matlab-base:latest'
    ]
    
    for image in base_images:
        exists = check_docker_image(image)
        status = "✅ Built" if exists else "❌ Not built"
        print(f"  {image}: {status}")
    
    # Check test videos
    print("\n🎬 Test Videos:")
    test_videos_dir = base_dir / 'data' / 'test_videos'
    if test_videos_dir.exists():
        videos = list(test_videos_dir.glob('*.mp4'))
        print(f"  Original videos: {len([v for v in videos if 'degraded' not in str(v)])}")
        print(f"  Degraded videos: {len([v for v in videos if 'degraded' in str(v)])}")
    else:
        print("  ❌ Test videos directory not found")
    
    # Check metric implementations
    print("\n📊 Metric Implementations:")
    metrics_status = []
    
    for metric_name, metric_dir_name in metrics:
        metric_dir = base_dir / 'metrics' / metric_dir_name
        status = check_metric_implementation(metric_dir_name, metric_dir)
        metrics_status.append(status)
        
        # Print status
        if status['status'] == 'implemented':
            emoji = "✅"
        elif status['status'] == 'in_progress':
            emoji = "🔄"
        else:
            emoji = "❌"
        
        docker_status = "🐳" if status['docker'] else "📦"
        print(f"  {emoji} {metric_name}: {status['status']} {docker_status}")
    
    # Summary statistics
    implemented = sum(1 for m in metrics_status if m['status'] == 'implemented')
    in_progress = sum(1 for m in metrics_status if m['status'] == 'in_progress')
    not_started = sum(1 for m in metrics_status if m['status'] == 'not_started')
    
    print(f"\n📈 Summary:")
    print(f"  Total metrics: {len(metrics)}")
    print(f"  ✅ Implemented: {implemented} ({implemented/len(metrics)*100:.1f}%)")
    print(f"  🔄 In progress: {in_progress} ({in_progress/len(metrics)*100:.1f}%)")
    print(f"  ❌ Not started: {not_started} ({not_started/len(metrics)*100:.1f}%)")
    
    # Check orchestration components
    print(f"\n🎯 System Components:")
    components = {
        'Orchestrator': base_dir / 'orchestrator' / 'orchestrator.py',
        'Report Generator': base_dir / 'reporting' / 'report_generator.py',
        'Main Runner': base_dir / 'run_vqa_system.py'
    }
    
    for component_name, component_path in components.items():
        exists, size = check_file_exists(component_path)
        status = "✅ Implemented" if exists else "❌ Not implemented"
        print(f"  {component_name}: {status}")
    
    # Write detailed status to JSON
    detailed_status = {
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics_status,
        'summary': {
            'total': len(metrics),
            'implemented': implemented,
            'in_progress': in_progress,
            'not_started': not_started
        }
    }
    
    status_file = base_dir / 'implementation_status.json'
    with open(status_file, 'w') as f:
        json.dump(detailed_status, f, indent=2)
    
    print(f"\n📄 Detailed status written to: {status_file}")
    
    # Next steps
    print("\n🚀 Next Steps:")
    if not_started > 0:
        print(f"  1. Complete implementation of {not_started} remaining metrics")
    if not all(check_docker_image(img) for img in base_images):
        print(f"  2. Build base Docker images: ./build_base_images.sh all")
    if not (base_dir / 'data' / 'test_videos').exists():
        print(f"  3. Download test videos: ./complete_todo2.sh")
    print(f"  4. Test implemented metrics individually")
    print(f"  5. Create orchestration system")
    print(f"  6. Create report generation system")
    print(f"  7. Run complete system validation")


if __name__ == '__main__':
    main()