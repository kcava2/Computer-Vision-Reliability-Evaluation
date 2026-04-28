"""
reliability_test.py — Entry point for the MRS evaluation framework.

Loads each trained model from the REGISTRY, runs the full reliability
evaluation against three dataset variants (baseline, shifted, perturbed),
prints a formatted report, and saves results to reliability_results.json.

Models without a saved checkpoint are skipped gracefully with a warning.
Only AlexNet has trained weights by default; ResNet50 and VGG16 are skipped
until their checkpoints are available.
"""

import os
import sys
import json
import warnings
import importlib.util
from datetime import datetime

import torch

warnings.filterwarnings('ignore', category=DeprecationWarning, module='torchvision')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'trained models')

RELIABILITY_DIR = os.path.join(PROJECT_ROOT, 'reliability backend')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODELS_DIR)
sys.path.insert(0, RELIABILITY_DIR)

from reliability_datasets import get_reliability_loaders
from reliability_metrics import ReliabilityEvaluator

# Mirror evaluation.py REGISTRY exactly so model loading is consistent
REGISTRY = [
    {
        'name': 'AlexNet',
        'file': os.path.join(MODELS_DIR, 'alexnet.py'),
        'class': 'AlexNet',
        'checkpoint': os.path.join(MODELS_DIR, 'alexnet.pth'),
    },
    {
        'name': 'VGG16',
        'file': os.path.join(MODELS_DIR, 'vggnet.py'),
        'class': 'VGG16',
        'checkpoint': os.path.join(MODELS_DIR, 'vgg16.pth'),
    },
    {
        'name': 'ResNet50',
        'file': os.path.join(MODELS_DIR, 'resnet.py'),
        'class': 'ResNet50',
        'checkpoint': os.path.join(MODELS_DIR, 'resnet50.pth'),
    },
]


def load_model_class(name, file_path, class_name):
    """Dynamically import a model class from its source file (mirrors evaluation.py)."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _make_json_safe(obj):
    """Recursively convert numpy/torch scalars to Python native types for JSON output."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_make_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, float) and (obj != obj):  # nan check
        return None
    return obj


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Build three dataloaders once — shared across all models
    print('\nPreparing reliability dataloaders ...')
    loader_D, loader_Dprime, loader_Dtilde, dataset_info = get_reliability_loaders(
        data_dir=os.path.join(PROJECT_ROOT, 'data'),
        batch_size=128,
        num_workers=0,
    )
    print(f'  Baseline D       : {len(loader_D.dataset):,} images')
    print(f'  Shifted D_prime  : {len(loader_Dprime.dataset):,} images')
    print(f'  Perturbed D_tilde: {len(loader_Dtilde.dataset):,} images')
    print(f'\n  D_tilde transform distribution:')
    for name, count in sorted(dataset_info['transform_dist'].items()):
        print(f'    {name:<25}: {count:,}')

    all_results = {}

    for cfg in REGISTRY:
        print(f'\n{"="*65}')
        print(f'  Evaluating: {cfg["name"]}')
        print(f'{"="*65}')

        if not os.path.exists(cfg['checkpoint']):
            print(f'  WARNING: Checkpoint not found at {cfg["checkpoint"]}')
            print(f'  Skipping {cfg["name"]} — train it first to generate weights.\n')
            continue

        # Load model using same pattern as evaluation.py
        ModelClass = load_model_class(cfg['name'], cfg['file'], cfg['class'])
        model = ModelClass(num_classes=100).to(device)
        model.load_state_dict(
            torch.load(cfg['checkpoint'], map_location=device, weights_only=True)
        )
        print(f'  Loaded checkpoint: {os.path.basename(cfg["checkpoint"])}')

        evaluator = ReliabilityEvaluator(model, device, config=None)
        results = evaluator.compute_all_metrics(loader_D, loader_Dprime, loader_Dtilde, dataset_info=dataset_info)
        evaluator.print_report(results, model_name=cfg['name'])

        all_results[cfg['name']] = results

    # ------------------------------------------------------------------
    # Save to JSON
    # ------------------------------------------------------------------
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'device': str(device),
            'dataset': 'CIFAR-100 (train split, seed=42, 10000 test samples)',
            'framework': 'Model Reliability Score (MRS)',
        },
        'models': _make_json_safe(all_results),
    }

    out_path = os.path.join(PROJECT_ROOT, 'reliability_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nResults saved to: {out_path}')

    if not all_results:
        print('\nNo models were evaluated (no checkpoints found).')
        print('Train at least one model and re-run reliability_test.py.')
