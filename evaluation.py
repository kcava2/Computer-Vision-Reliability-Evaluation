import os
import sys
import warnings
import importlib.util
import torch

# Suppress a numpy 2.4 incompatibility inside torchvision's CIFAR pickle loader
warnings.filterwarnings('ignore', category=DeprecationWarning, module='torchvision')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'trained models')

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, MODELS_DIR)

from dataloader import get_loaders

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
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    accuracy = 100.0 * correct / total
    return accuracy, all_preds, all_labels


def compute_f1_macro(preds, labels, num_classes=100):
    preds_t = torch.tensor(preds)
    labels_t = torch.tensor(labels)

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for p, l in zip(preds_t, labels_t):
        confusion[l, p] += 1

    f1_scores = []
    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    _, test_loader = get_loaders()
    print(f'Test set: {len(test_loader.dataset)} images\n')

    for cfg in REGISTRY:
        print(f'--- {cfg["name"]} ---')

        if not os.path.exists(cfg['checkpoint']):
            print(f'  Checkpoint not found: {cfg["checkpoint"]}')
            print(f'  Skipping — run {os.path.basename(cfg["file"])} to train first.\n')
            continue

        ModelClass = load_model_class(cfg['name'], cfg['file'], cfg['class'])
        model = ModelClass(num_classes=100).to(device)
        model.load_state_dict(torch.load(cfg['checkpoint'], map_location=device, weights_only=True))

        accuracy, preds, labels = evaluate_model(model, test_loader, device)
        f1 = compute_f1_macro(preds, labels)

        print(f'  Accuracy : {accuracy:.2f}%')
        print(f'  F1 Score : {f1:.4f}  (macro)\n')
