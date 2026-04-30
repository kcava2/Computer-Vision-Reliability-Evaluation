"""
reliability_datasets.py — Dataset variants for MRS evaluation.

Three loaders over the same 10K CIFAR-100 test indices:
  D        — baseline clean test set (matches evaluation.py exactly)
  D_prime  — FFT low-frequency reconstruction (distribution shift)
  D_tilde  — stochastic per-image degradation (4 transforms + 6 pairwise combos)
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset


MEAN = [0.5071, 0.4867, 0.4408]
STD  = [0.2675, 0.2565, 0.2761]

_BASE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

_TRANSFORM_NAMES = [
    'gaussian',
    'motion_blur',
    'brightness',
    'rotation',
    'gaussian+motion_blur',
    'gaussian+brightness',
    'gaussian+rotation',
    'motion_blur+brightness',
    'motion_blur+rotation',
    'brightness+rotation',
]


def _get_test_indices(data_dir, seed=42, train_ratio=0.8):
    """Reproduce the exact 10K test indices that dataloader.py produces."""
    dummy = datasets.CIFAR100(root=data_dir, train=True, download=False)
    total = len(dummy)
    gen = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(total, generator=gen).tolist()
    split = int(train_ratio * total)
    return all_indices[split:]


def fft_low_frequency_reconstruction(tensor, radius=8):
    """Keep only the low-frequency circular region of the 2D FFT spectrum.

    Args:
        tensor: Normalized image tensor of shape (C, H, W).
        radius: Radius of the low-pass circular mask in frequency space.

    Returns:
        Reconstructed tensor with same shape; high-frequency content zeroed.
    """
    C, H, W = tensor.shape
    f = torch.fft.fft2(tensor)
    f_shifted = torch.fft.fftshift(f, dim=(-2, -1))

    cy, cx = H // 2, W // 2
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = (dist <= radius).float()  # (H, W)

    f_filtered = f_shifted * mask.unsqueeze(0)  # broadcast over C
    f_unshifted = torch.fft.ifftshift(f_filtered, dim=(-2, -1))
    return torch.fft.ifft2(f_unshifted).real


def _apply_gaussian(tensor, sigma, rng):
    noise = torch.from_numpy(rng.normal(0.0, sigma, tensor.shape).astype(np.float32))
    return tensor + noise


def _apply_motion_blur(tensor, kernel_size=5):
    C = tensor.shape[0]
    kernel = torch.ones(C, 1, 1, kernel_size) / kernel_size  # depthwise horizontal
    return F.conv2d(tensor.unsqueeze(0), kernel, padding=(0, kernel_size // 2), groups=C).squeeze(0)


def _apply_brightness(tensor, factor=0.5):
    return tensor * factor


def _apply_rotation(tensor, angle=15.0):
    return TF.rotate(tensor, angle)


class StochasticDegradationDataset(Dataset):
    """Per-image stochastic degradation over a CIFAR-100 base dataset.

    Assigns one of 10 transforms (4 single + 6 pairwise combos) to each image
    at construction time using a seeded RNG.  Assignments are fixed so the
    dataset is fully deterministic across runs.
    """

    def __init__(self, base_cifar, indices, seed=42, sigma=0.1):
        self.base_cifar = base_cifar
        self.indices = indices
        self.sigma = sigma
        N = len(indices)
        rng = np.random.default_rng(seed)
        self.transform_assignments = rng.integers(0, len(_TRANSFORM_NAMES), size=N)
        self._noise_seeds = rng.integers(0, 2 ** 31, size=N)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_cifar[self.indices[idx]]
        name = _TRANSFORM_NAMES[self.transform_assignments[idx]]
        rng = np.random.default_rng(int(self._noise_seeds[idx]))
        for part in name.split('+'):
            if part == 'gaussian':
                img = _apply_gaussian(img, self.sigma, rng)
            elif part == 'motion_blur':
                img = _apply_motion_blur(img)
            elif part == 'brightness':
                img = _apply_brightness(img)
            elif part == 'rotation':
                img = _apply_rotation(img)
        return img, label

    @property
    def transform_log(self):
        """List of transform names, one entry per image."""
        return [_TRANSFORM_NAMES[i] for i in self.transform_assignments]

    @property
    def transform_dist(self):
        """Dict mapping transform_name -> count."""
        counts = {}
        for i in self.transform_assignments:
            name = _TRANSFORM_NAMES[i]
            counts[name] = counts.get(name, 0) + 1
        return dict(counts)


class FFTReconstructionDataset(Dataset):
    """CIFAR-100 subset where each image is replaced by its FFT low-pass reconstruction.

    Simulates distribution shift by removing high-frequency content.
    """

    def __init__(self, base_cifar, indices, radius=12):
        self.base_cifar = base_cifar
        self.indices = indices
        self.radius = radius

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.base_cifar[self.indices[idx]]
        return fft_low_frequency_reconstruction(img, radius=self.radius), label


def _compute_dist_stats(loader_D, loader_Dprime, radius=8, max_batches=5):
    """Compare pixel statistics and spectral energy between D and D_prime.

    Samples up to max_batches batches from each loader.
    """

    def _batch_stats(batch):
        B, C, H, W = batch.shape
        f = torch.fft.fft2(batch)
        f_s = torch.fft.fftshift(f, dim=(-2, -1))
        cy, cx = H // 2, W // 2
        ys = torch.arange(H, dtype=torch.float32)
        xs = torch.arange(W, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        dist = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        hf_mask = (dist > radius).float()
        hf_energy = float((f_s.abs() ** 2 * hf_mask.unsqueeze(0).unsqueeze(0)).sum())
        total_energy = float((f_s.abs() ** 2).sum())
        return float(batch.mean()), float(batch.std()), hf_energy, total_energy

    stats = {}
    for key, loader in (('D', loader_D), ('D_prime', loader_Dprime)):
        means, stds, hf_total, e_total = [], [], 0.0, 0.0
        for i, (batch, _) in enumerate(loader):
            if i >= max_batches:
                break
            m, s, hf, e = _batch_stats(batch)
            means.append(m)
            stds.append(s)
            hf_total += hf
            e_total += e
        stats[key] = {
            'mean': float(np.mean(means)),
            'std': float(np.mean(stds)),
            'hf_energy_ratio': float(hf_total / e_total) if e_total > 0 else 0.0,
        }
    return stats


def get_reliability_loaders(
    data_dir='./data',
    batch_size=128,
    num_workers=0,
    seed=42,
    train_ratio=0.8,
    sigma=0.1,
    fft_radius=12,
):
    """Return three dataloaders and dataset metadata for MRS evaluation.

    Returns:
        loader_D       — baseline clean test set
        loader_Dprime  — FFT low-frequency reconstructed (distribution shift)
        loader_Dtilde  — stochastic per-image degradation
        dataset_info   — dict with transform_log, transform_dist, dist_stats
    """
    test_indices = _get_test_indices(data_dir, seed=seed, train_ratio=train_ratio)

    base_cifar = datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=_BASE_TRANSFORM
    )

    dataset_D = Subset(base_cifar, test_indices)
    dataset_Dprime = FFTReconstructionDataset(base_cifar, test_indices, radius=fft_radius)
    dataset_Dtilde = StochasticDegradationDataset(base_cifar, test_indices, seed=seed, sigma=sigma)

    def make_loader(ds):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    loader_D = make_loader(dataset_D)
    loader_Dprime = make_loader(dataset_Dprime)
    loader_Dtilde = make_loader(dataset_Dtilde)

    print('  Computing spectral distribution statistics ...')
    dist_stats = _compute_dist_stats(loader_D, loader_Dprime, radius=fft_radius)

    dataset_info = {
        'transform_log': dataset_Dtilde.transform_log,
        'transform_dist': dataset_Dtilde.transform_dist,
        'dist_stats': dist_stats,
    }

    return loader_D, loader_Dprime, loader_Dtilde, dataset_info


if __name__ == '__main__':
    loader_D, loader_Dprime, loader_Dtilde, info = get_reliability_loaders()
    print(f'Baseline D       : {len(loader_D.dataset)} images')
    print(f'Shifted D_prime  : {len(loader_Dprime.dataset)} images')
    print(f'Perturbed D_tilde: {len(loader_Dtilde.dataset)} images')
    print('\nTransform distribution:')
    for name, count in sorted(info['transform_dist'].items()):
        print(f'  {name:<25}: {count}')
    print('\nSpectral statistics:')
    for split, s in info['dist_stats'].items():
        print(f'  {split}: mean={s["mean"]:.4f}  std={s["std"]:.4f}  HF_ratio={s["hf_energy_ratio"]:.6f}')
