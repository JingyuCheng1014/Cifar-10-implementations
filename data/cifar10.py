import torch
from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

'''
Method 1: Use torchvision's own CIFAR10 + reasonable data enhancement + DataLoader
'''
def get_dataloaders(cfg):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = CIFAR10(root=cfg.data_root, train=True, download=True, transform=train_transform)
    val_ds   = CIFAR10(root=cfg.data_root, train=False, download=True, transform=val_transform)

    # 给 DataLoader 一个固定随机数发生器
    g = torch.Generator()
    g.manual_seed(8539)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    return train_loader, val_loader




'''
Method 2: Writing your own dataclass
'''
# class CIFAR10_Dataset(Dataset):
#     def __init__(self, root_dir: str, train: bool = True, transform=None):
#         self.root_dir = root_dir
#         self.train = train
#         self.transform = transform
#         cifar = CIFAR10(root=self.root_dir, train=self.train, download=True)
#         self.images = cifar.data
#         self.labels = cifar.targets
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img = Image.fromarray(self.images[idx])
#         label = self.labels[idx]
#         img = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)
#         return img, label
#
#
# def get_dataloaders(cfg):
#     train_t = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465),
#                              (0.2470, 0.2435, 0.2616)),
#     ])
#     val_t = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465),
#                              (0.2470, 0.2435, 0.2616)),
#     ])
#
#     train_ds = CIFAR10_Dataset(cfg.data_root, train=True,  transform=train_t)
#     val_ds   = CIFAR10_Dataset(cfg.data_root, train=False, transform=val_t)
#
#     train_loader = DataLoader(
#         train_ds, batch_size=cfg.batch_size, shuffle=True,
#         num_workers=cfg.num_workers, pin_memory=True,
#         persistent_workers=(cfg.num_workers > 0),
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=cfg.batch_size, shuffle=False,
#         num_workers=cfg.num_workers, pin_memory=True,
#         persistent_workers=(cfg.num_workers > 0),
#     )
#     return train_loader, val_loader
