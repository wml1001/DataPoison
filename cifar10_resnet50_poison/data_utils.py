# data_utils.py
import torch
from torch.utils.data import Dataset, DataLoader,Sampler
from torchvision import datasets, transforms
import config
from torchvision.transforms import ToPILImage

# data_utils.py
import random

class PoisonedBatchSampler(Sampler):
    """
    BatchSampler: 每批包含固定比例的投毒样本和干净样本。
    """
    def __init__(self, poison_indices, total_size, batch_size, poison_ratio=0.1, drop_last=False):
        self.poison_idx = list(poison_indices)
        self.clean_idx = [i for i in range(total_size) if i not in poison_indices]
        self.batch_size = batch_size
        self.n_poison = int(batch_size * poison_ratio)
        self.drop_last = drop_last
        self.total_batches = total_size // batch_size

    def __iter__(self):
        for _ in range(self.total_batches):
            batch = []
            # 随机选投毒样本
            if len(self.poison_idx) >= self.n_poison:
                batch = random.sample(self.poison_idx, self.n_poison)
            else:
                batch = random.choices(self.poison_idx, k=self.n_poison)
            # 补充干净样本
            need = self.batch_size - len(batch)
            batch += random.sample(self.clean_idx, need)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.total_batches


def get_raw_cifar10(train=True):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root=config.DATA_ROOT, train=train, download=True, transform=transform)
    return dataset

def _make_resize(size):
    try:
        return transforms.Resize(size, antialias=True)
    except TypeError:
        return transforms.Resize(size)

def get_transform(train=True, to_tensor=True):
    tr = []
    if train:
        tr.extend([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
    tr.append(_make_resize(224))
    if to_tensor:
        tr.append(transforms.ToTensor())
    tr.append(transforms.Normalize(config.NORM_MEAN, config.NORM_STD))
    return transforms.Compose(tr)

class PoisonedCIFAR10(Dataset):
    def __init__(self, base_dataset, poison_indices, poison_samples, clean_transform, poison_transform):
        self.base_dataset = base_dataset
        self.poison_indices = set(poison_indices)
        self.poison_samples = poison_samples.detach().cpu()
        self.clean_transform = clean_transform
        self.poison_transform = poison_transform
        self.idx_to_poison = {idx: i for i, idx in enumerate(poison_indices)}
        self.to_pil = ToPILImage()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        is_poison = idx in self.poison_indices
        if is_poison:
            poison_idx = self.idx_to_poison[idx]
            img = self.poison_samples[poison_idx]
        transform = self.poison_transform if is_poison else self.clean_transform
        if transform is not None:
            try:
                img = transform(img)
            except Exception:
                img = self.to_pil(img)
                img = transform(img)
        return img, label

def get_poisoned_dataloader(base_dataset, poison_indices, poison_samples, batch_size, shuffle=True, num_workers=4):
    clean_transform = get_transform(train=True, to_tensor=False)
    poison_transform = get_transform(train=False, to_tensor=False)
    poisoned_dataset = PoisonedCIFAR10(base_dataset, poison_indices, poison_samples, clean_transform, poison_transform)
    loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return loader

def get_clean_dataloaders(num_workers=4):
    train_transform = get_transform(train=True, to_tensor=True)
    test_transform = get_transform(train=False, to_tensor=True)
    train_dataset = datasets.CIFAR10(root=config.DATA_ROOT, train=True, download=True, transform=train_transform)
    test_dataset  = datasets.CIFAR10(root=config.DATA_ROOT, train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader