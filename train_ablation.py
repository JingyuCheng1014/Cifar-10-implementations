import os, time, csv
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
    epochs: int = 50  # 可改
    lr: float = 3e-4  # 学习率，优化器更新时的步长
    weight_decay: float = 5e-2
    warmup_epochs: int = 5  # 线性 warmup 轮数（0 关闭）
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


class ViT(nn.Module):
    def __init__(self, patch_size=4, dim=128, depth=6, heads=8, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        num_patches = (32 // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=depth
        )

        # Classification head
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Patch embedding
        patches = self.patch_embed(x)  # [B, dim, H/patch_size, W/patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, dim]

        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Classification
        x = x[:, 0]  # Take cls token
        return self.head(x)

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


def run_ablation(train_loader, val_loader, device, cfg, patch_size=4, dim=128, depth=6, heads=8, tag_prefix=""):
    """训练一个配置，返回 best_val_acc, sec/epoch, params，并将 best checkpoint 与曲线保存到磁盘"""
    # 合法性检查（避免 heads 不整除）
    assert dim % heads == 0, f"d_model {dim} must be divisible by nhead {heads}"

    config_name = f"{tag_prefix}ps{patch_size}_dim{dim}_d{depth}_h{heads}"
    print(f"\n[Run] {config_name}")

    model = ViT(patch_size=patch_size, dim=dim, depth=depth, heads=heads, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    history = []
    best_val_acc, best_state = 0.0, None

    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        # 简单 warmup
        if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
            warmup_lr = cfg.lr * float(epoch + 1) / float(cfg.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append((epoch + 1, tr_loss, tr_acc, val_loss, val_acc))

        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"Epoch [{epoch + 1}/{cfg.epochs}] "
              f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"BestVal={best_val_acc:.4f}")

    sec_per_epoch = (time.perf_counter() - t0) / cfg.epochs
    params = sum(p.numel() for p in model.parameters())

    # 保存 best ckpt 与曲线
    torch.save(best_state, f"checkpoints/vit_{config_name}_best.pth")
    with open(f"logs/{config_name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        w.writerows(history)

    print(f"  ✓ Saved: checkpoints/vit_{config_name}_best.pth | logs/{config_name}.csv")
    print(f"  ✓ Params: {params / 1e6:.2f}M | sec/epoch: {sec_per_epoch:.3f}")
    return best_val_acc, sec_per_epoch, params



# =========================
# 6) 主程序
# =========================
def main(cfg: Cfg):
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(cfg)
    classes = train_loader.dataset.classes  # CIFAR10 类别名列表

    # 可视化一个 batch（来自训练集）
    # visualize_one_batch(train_loader, classes)  # 可视化只在第一次运行时需要

    base_cfg = dict(patch_size=4, dim=128, depth=6, heads=8)
    print("\n=== Baseline ===")
    base_acc, base_sec, base_params = run_ablation(train_loader, val_loader, device, cfg, **base_cfg, tag_prefix="base_")
    print(f"Baseline Acc={base_acc:.4f}, Params={base_params/1e6:.2f}M, sec/epoch={base_sec:.3f}")

    # 默认参数
    default_patch_size, default_dim, default_depth, default_heads = 4, 128, 6, 8

    results = []

    # Patch Size Ablation
    print("\n=== Patch Size Ablation ===")
    best_patch = None
    for patch in [2, 4, 8]:
        acc, sec, params = run_ablation(train_loader, val_loader, device, cfg,
                                        patch_size=patch, dim=base_cfg["dim"], depth=base_cfg["depth"],
                                        heads=base_cfg["heads"],
                                        tag_prefix="patch_")
        results.append(("patch", patch, acc, sec, params))
        if (best_patch is None) or (acc > best_patch[1]):
            best_patch = (patch, acc)

    # ---- Embedding Dim ----
    print("\n=== Embedding Dimension Ablation ===")
    best_dim = None
    for dim in [96, 128, 256]:
        # 确保可整除
        heads = base_cfg["heads"]
        if dim % heads != 0:
            continue
        acc, sec, params = run_ablation(train_loader, val_loader, device, cfg,
                                        patch_size=best_patch[0] if best_patch else base_cfg["patch_size"],
                                        dim=dim, depth=base_cfg["depth"], heads=heads,
                                        tag_prefix="dim_")
        results.append(("dim", dim, acc, sec, params))
        if (best_dim is None) or (acc > best_dim[1]):
            best_dim = (dim, acc)

    # ---- Depth ----
    print("\n=== Depth Ablation ===")
    best_depth = None
    for depth in [4, 6, 12]:
        acc, sec, params = run_ablation(train_loader, val_loader, device, cfg,
                                        patch_size=best_patch[0] if best_patch else base_cfg["patch_size"],
                                        dim=best_dim[0] if best_dim else base_cfg["dim"],
                                        depth=depth, heads=base_cfg["heads"],
                                        tag_prefix="depth_")
        results.append(("depth", depth, acc, sec, params))
        if (best_depth is None) or (acc > best_depth[1]):
            best_depth = (depth, acc)

    # ---- Heads ----
    print("\n=== Heads Ablation ===")
    best_heads = None
    # 与 dim=128 兼容的 heads
    for heads in [1, 2, 4, 8, 16]:
        dim = best_dim[0] if best_dim else base_cfg["dim"]
        if dim % heads != 0:
            continue
        acc, sec, params = run_ablation(train_loader, val_loader, device, cfg,
                                        patch_size=best_patch[0] if best_patch else base_cfg["patch_size"],
                                        dim=dim, depth=best_depth[0] if best_depth else base_cfg["depth"],
                                        heads=heads, tag_prefix="heads_")
        results.append(("heads", heads, acc, sec, params))
        if (best_heads is None) or (acc > best_heads[1]):
            best_heads = (heads, acc)

    # ---- 汇总打印 ----
    print("\n=== Ablation Results (best val acc per config) ===")
    for axis, val, acc, sec, params in results:
        print(f"{axis:>6}: {val:>4} | acc={acc:.4f} | params={params / 1e6:.2f}M | sec/epoch={sec:.3f}")

    print("\nBest by axis:")
    print(f"  Patch Size: {best_patch[0]} (acc={best_patch[1]:.4f})")
    print(f"  Dim       : {best_dim[0]} (acc={best_dim[1]:.4f})")
    print(f"  Depth     : {best_depth[0]} (acc={best_depth[1]:.4f})")
    print(f"  Heads     : {best_heads[0]} (acc={best_heads[1]:.4f})")

    # ---- 最佳组合再训 ----
    print("\n=== Final Best-Combo Training ===")
    final_cfg = dict(
        patch_size=best_patch[0],
        dim=best_dim[0],
        depth=best_depth[0],
        heads=best_heads[0],
    )
    final_acc, final_sec, final_params = run_ablation(train_loader, val_loader, device, cfg, **final_cfg,
                                                      tag_prefix="final_")
    print(
        f"\nFinal Combo: {final_cfg} | acc={final_acc:.4f} | params={final_params / 1e6:.2f}M | sec/epoch={final_sec:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main(cfg)
