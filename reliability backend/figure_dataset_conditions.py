"""
figure_dataset_conditions.py — Dataset condition visualization grid.

4 rows (representative CIFAR-100 images) x 12 columns (conditions):
  Clean | FFT (r=12) | 10 stochastic degradation transforms.

Outputs: figures/dataset_conditions.png
"""

import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit('matplotlib is required: pip install matplotlib')

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
DATA_DIR    = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(FIGURES_DIR, exist_ok=True)

from reliability_datasets import (
    fft_low_frequency_reconstruction,
    MEAN, STD,
    _apply_gaussian, _apply_motion_blur, _apply_brightness, _apply_rotation,
)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# CIFAR-100 fine labels (alphabetical, indices 0-99)
_CLASSES = [
    'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear',
    'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm',
]

# 4 visually distinct target classes: bicycle, forest, orchid, tiger
TARGET_CLASSES = [8, 33, 54, 88]

COL_LABELS = [
    'Clean', 'FFT\n(r=12)',
    'Gaussian\nnoise', 'Motion\nblur', 'Brightness\n-50%', 'Rotation\n15 deg',
    'Gauss +\nMotion', 'Gauss +\nBright', 'Gauss +\nRot',
    'Motion +\nBright', 'Motion +\nRot', 'Bright +\nRot',
]

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

_MEAN_T = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
_STD_T  = torch.tensor(STD,  dtype=torch.float32).view(3, 1, 1)


def _norm(t):
    return (t - _MEAN_T) / _STD_T


def _to_display(t):
    """Unnormalize and convert to clipped [H,W,3] numpy float32."""
    return torch.clamp(t * _STD_T + _MEAN_T, 0.0, 1.0).permute(1, 2, 0).numpy()


def get_cells(raw, rng_seed):
    """Return 12 numpy images [H,W,3] in [0,1] for one input raw tensor."""
    norm = _norm(raw)
    s = int(rng_seed)
    return [
        raw.permute(1, 2, 0).numpy(),                                                       # Clean
        _to_display(fft_low_frequency_reconstruction(norm, radius=12)),                      # FFT
        _to_display(_apply_gaussian(norm, 0.1, np.random.default_rng(s))),                  # Gaussian
        _to_display(_apply_motion_blur(norm)),                                               # Motion blur
        _to_display(_apply_brightness(norm)),                                                # Brightness
        _to_display(_apply_rotation(norm)),                                                  # Rotation
        _to_display(_apply_motion_blur(                                                      # G + Motion
            _apply_gaussian(norm, 0.1, np.random.default_rng(s)))),
        _to_display(_apply_brightness(                                                       # G + Bright
            _apply_gaussian(norm, 0.1, np.random.default_rng(s)))),
        _to_display(_apply_rotation(                                                         # G + Rot
            _apply_gaussian(norm, 0.1, np.random.default_rng(s)))),
        _to_display(_apply_brightness(_apply_motion_blur(norm))),                            # M + Bright
        _to_display(_apply_rotation(_apply_motion_blur(norm))),                              # M + Rot
        _to_display(_apply_rotation(_apply_brightness(norm))),                               # B + Rot
    ]


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == '__main__':
    display_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),   # [0,1], no normalization
    ])
    cifar = datasets.CIFAR100(root=DATA_DIR, train=True, download=False,
                               transform=display_tf)

    # Find first occurrence of each target class
    selected = {}
    for i, lbl in enumerate(cifar.targets):
        if lbl in TARGET_CLASSES and lbl not in selected:
            selected[lbl] = i
        if len(selected) == len(TARGET_CLASSES):
            break

    if len(selected) < len(TARGET_CLASSES):
        missing = set(TARGET_CLASSES) - set(selected)
        raise RuntimeError(f'Could not find images for classes: {missing}')

    row_indices = [selected[c] for c in TARGET_CLASSES]
    n_rows = len(row_indices)
    n_cols = len(COL_LABELS)

    cell_px = 1.55   # inches per cell
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_px, n_rows * cell_px + 0.6),
    )
    fig.subplots_adjust(hspace=0.04, wspace=0.04)

    for r, idx in enumerate(row_indices):
        raw, label = cifar[idx]
        class_name = _CLASSES[label]
        cells = get_cells(raw, rng_seed=idx)

        for c, img in enumerate(cells):
            ax = axes[r, c]
            ax.imshow(img, interpolation='bilinear')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

            if r == 0:
                ax.set_title(COL_LABELS[c], fontsize=7.5, pad=3)
            if c == 0:
                ax.set_ylabel(class_name, fontsize=9, labelpad=4)

    # Column group separators: thin vertical lines between D, D', D~
    # D = col 0, D' = col 1, D~ = cols 2-11
    for ax in axes[:, 1]:
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['left'].set_color('#555555')
    for ax in axes[:, 2]:
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['left'].set_color('#555555')

    fig.suptitle(
        "Dataset Conditions — D (clean), D' (FFT-shifted), D~ (stochastic degradation)",
        fontsize=11, y=1.005,
    )

    out = os.path.join(FIGURES_DIR, 'dataset_conditions.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {os.path.basename(out)}  (300 dpi)')
    print(f'Classes shown: {[_CLASSES[c] for c in TARGET_CLASSES]}')
