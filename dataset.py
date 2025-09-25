# dataset.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 增加颜色抖动
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def build_loader(root, batch_size, num_workers):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    train_ds = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ld, val_ld, train_ds.classes