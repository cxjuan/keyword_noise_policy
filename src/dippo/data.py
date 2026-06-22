from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None, recursive=True):
        self.root = Path(root)
        self.files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.JPEG', '*.JPG']:
            self.files.extend(sorted(self.root.rglob(ext) if recursive else self.root.glob(ext)))
        if len(self.files) == 0:
            raise RuntimeError(f'No image files found under {root}')
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def make_loader(name: str, root: str, batch_size: int, train: bool = True, download: bool = False):
    name = name.lower()
    tfm = transforms.Compose([transforms.ToTensor()])
    if name in ['mnist', 'MNIST']:
        ds = datasets.MNIST(root=root, train=train, download=False, transform=tfm)
    elif name in ['fashion-mnist', 'fashionmnist', 'fmnist', 'FashionMNIST']:
        ds = datasets.FashionMNIST(root=root, train=train, download=False, transform=tfm)
    elif name in ['cifar10', 'CIFAR10']:
        ds = datasets.CIFAR10(root=root, train=train, download=False, transform=tfm)
    elif name in ['imagefolder', 'celeba32', 'imagenet32', 'ImageNet', 'ImageNet64', 'ImageNet32', 'celeba', 'celeba32',
                  'celeba64', 'CelebA']:
        ds = FlatImageFolder(root=root, transform=tfm, recursive=True)
    else:
        raise ValueError(f'Unsupported dataset: {name}')
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=2, pin_memory=True)
