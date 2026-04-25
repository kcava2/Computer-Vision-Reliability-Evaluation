import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_loaders(data_dir='./data', batch_size=128, num_workers=0, train_ratio=0.8, seed=42):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    # Load training split twice so each subset can have its own transform
    train_full = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_full = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=test_transform)

    total = len(train_full)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    split = int(train_ratio * total)

    train_loader = DataLoader(
        Subset(train_full, indices[:split]), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        Subset(test_full, indices[split:]), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False,
    )

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_loaders()
    print(f'Train batches : {len(train_loader)}  ({len(train_loader.dataset)} images)')
    print(f'Test batches  : {len(test_loader)}  ({len(test_loader.dataset)} images)')
    print('Dataloader ready.')
