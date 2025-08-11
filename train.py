import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.cifar10 import get_dataloaders
from model.vit import ViT


# =========================
# 1) 配置
# =========================
@dataclass
class Cfg:
    data_root: str
    batch_size: int
    num_workers: int
    # 训练超参
    epochs: int = 2
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Cfg(
    data_root="./data",
    batch_size=128,
    num_workers=4,
)


# =========================
# 2) 仅用于可视化的均值 / 方差
#    （数据标准化已经在 data/cifar10.py 里做了）
# =========================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


# =========================
# 4) 可视化（含反归一化）
# =========================
def denormalize(img_tensor, mean=CIFAR_MEAN, std=CIFAR_STD):
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    mean = torch.tensor(mean, device=img_tensor.device).view(1, 3, 1, 1)
    std  = torch.tensor(std,  device=img_tensor.device).view(1, 3, 1, 1)
    return img_tensor * std + mean

def visualize_one_batch(loader, classes):
    imgs, labels = next(iter(loader))
    grid = vutils.make_grid(imgs[:32], nrow=8, padding=2)
    grid = denormalize(grid)
    grid = grid.squeeze(0)
    npimg = grid.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(npimg)
    plt.axis('off')
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

    # ✅ 直接用 data/cifar10.py 的数据加载
    train_loader, val_loader = get_dataloaders(cfg)
    classes = train_loader.dataset.classes

    # 可视化一个 batch
    # visualize_one_batch(train_loader, classes)

    # ✨ 使用 ViT（来自 vit.py）
    # CIFAR-10: 32x32x3；选 patch_size=4（=> 8x8=64 个 patch）
    model = ViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=256,          # Transformer hidden size
        depth=6,          # Transformer layers
        heads=8,          # Multi-head attention heads
        mlp_dim=512,      # FFN hidden size
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        print(f"\nEpoch [{epoch+1}/{cfg.epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/vit_cifar10_best.pth")
            print(f"  ✓ Saved best model (acc={best_acc:.4f})")

    print("\nDone.")


if __name__ == "__main__":
    main(cfg)
