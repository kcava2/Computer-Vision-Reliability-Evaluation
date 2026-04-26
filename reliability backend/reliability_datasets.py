import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


class AddGaussianNoise:
    """Adds Gaussian noise to a tensor after normalization (sigma in normalized pixel space)."""

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.sigma


def _get_test_indices(data_dir, seed=42, train_ratio=0.8):
    """Reproduce the exact 10K test indices that dataloader.py produces.

    Must match dataloader.py lines 25-28 exactly so all three reliability
    loaders evaluate the same images as the baseline evaluation pipeline.
    """
    dummy = datasets.CIFAR100(root=data_dir, train=True, download=False)
    total = len(dummy)  # always 50000
    gen = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(total, generator=gen).tolist()
    split = int(train_ratio * total)  # 40000
    return all_indices[split:]  # 10000 test indices


def get_reliability_loaders(
    data_dir='./data',
    batch_size=128,
    num_workers=0,
    seed=42,
    train_ratio=0.8,
    sigma=0.1,
):
    """Return three dataloaders over the same 10K test images with different transforms.

    Returns:
        loader_D       — baseline clean test set (identical to evaluation.py)
        loader_Dprime  — distribution-shifted: color jitter + horizontal flip
        loader_Dtilde  — perturbed: Gaussian noise added after normalization
    """
    MEAN = [0.5071, 0.4867, 0.4408]
    STD = [0.2675, 0.2565, 0.2761]

    test_indices = _get_test_indices(data_dir, seed=seed, train_ratio=train_ratio)

    # D — baseline: same as evaluation.py test transform
    baseline_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # D' — shifted: color jitter and horizontal flip simulate distribution shift
    shifted_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # D_tilde — perturbed: Gaussian noise applied after normalization
    perturbed_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
        AddGaussianNoise(sigma=sigma),
    ])

    def make_loader(transform):
        full_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=transform
        )
        subset = Subset(full_dataset, test_indices)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,  # preserve sequential order for failure index tracking
            num_workers=num_workers,
            pin_memory=False,
        )

    return make_loader(baseline_transform), make_loader(shifted_transform), make_loader(perturbed_transform)


if __name__ == '__main__':
    loader_D, loader_Dprime, loader_Dtilde = get_reliability_loaders()
    print(f'Baseline D       : {len(loader_D.dataset)} images, {len(loader_D)} batches')
    print(f'Shifted D_prime  : {len(loader_Dprime.dataset)} images, {len(loader_Dprime)} batches')
    print(f'Perturbed D_tilde: {len(loader_Dtilde.dataset)} images, {len(loader_Dtilde)} batches')
    # Verify all three operate on the same indices
    assert len(loader_D.dataset) == len(loader_Dprime.dataset) == len(loader_Dtilde.dataset)
    print('All three loaders share the same 10K test indices. Ready.')
