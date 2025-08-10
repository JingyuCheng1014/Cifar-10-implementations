import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from tqdm import tqdm


# =========================
# 1) 配置
# =========================
@dataclass
class Cfg:
    data_root: str
    batch_size: int
    num_workers: int
    # 下面是训练用到的最小超参（可以改）
    epochs: int = 2
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Cfg(
    data_root="./data",
    batch_size=128,
    num_workers=4,
)


# =========================
# 2) DataLoader
# =========================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

'''
Method 1: Use torchvision's own CIFAR10 + reasonable data enhancement + DataLoader
'''
def get_dataloaders(cfg):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN,
                             CIFAR_STD),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN,
                             CIFAR_STD),
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
#         transforms.Normalize(CIFAR_MEAN,
#                              CIFAR_STD),
#     ])
#     val_t = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(CIFAR_MEAN,
#                              CIFAR_STD),
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






# =========================
# 3) 一个极简 CNN（足够演示）
#    想换 ViT 时，把这里替换为你的 ViT 模型即可
# =========================
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            # [B,3,32,32] -> [B,64,16,16]
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # [B,64,16,16] -> [B,128,8,8]
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # [B,128,8,8] -> [B,256,4,4]
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),      # [B,256,1,1]
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.net(x)                 # [B,256,1,1]
        x = x.flatten(1)                # [B,256]
        return self.fc(x)               # [B,10]


# =========================
# 4) 可视化一个 batch（含反归一化）
# =========================
def denormalize(img_tensor, mean=CIFAR_MEAN, std=CIFAR_STD):
    """img_tensor: [C,H,W] 或 [B,C,H,W]，返回反归一化后的同形状张量"""
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    mean = torch.tensor(mean, device=img_tensor.device).view(1,3,1,1)
    std  = torch.tensor(std,  device=img_tensor.device).view(1,3,1,1)
    return img_tensor * std + mean

def visualize_one_batch(loader, classes):
    imgs, labels = next(iter(loader))        # 取一个 batch
    # 做一个网格
    grid = vutils.make_grid(imgs[:32], nrow=8, padding=2)  # 取前 32 张拼 8×4
    grid = denormalize(grid)                 # 反归一化到 [0,1]

    # 转成 numpy 显示
    npimg = grid.clamp(0,1).permute(1,2,0).cpu().numpy()
    plt.figure(figsize=(10,5))
    plt.imshow(npimg)
    plt.axis('off')
    # 拼接前 8 个标签的类名（可按需调整）
    title = " | ".join([classes[labels[i].item()] for i in range(min(8, len(labels)))])
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
# 5) 训练 & 验证
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = logits.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += labels.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Eval ", leave=False)
    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        _, preds = logits.max(1)
        correct  += preds.eq(labels).sum().item()
        total    += labels.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss/total, correct/total







# =========================
# 6) 主程序
# =========================
def main(cfg: Cfg):
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(cfg)
    classes = train_loader.dataset.classes  # CIFAR10 类别名列表

    # 可视化一个 batch（来自训练集）
    visualize_one_batch(train_loader, classes)

    model = TinyCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        print(f"\nEpoch [{epoch+1}/{cfg.epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        # 简单保存最优
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/tinycnn_cifar10_best.pth")
            print(f"  ✓ Saved best model (acc={best_acc:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main(cfg)
