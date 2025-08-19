import os
import random
import numpy as np
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils as vutils
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from data.cifar10 import get_dataloaders

from model.swin import swin_t, swin_s, swin_b, swin_l
from config import cfg as cfg_from_config, VIT, VIT_CONVWIN, build_model



# =========================
# 1) 配置
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="vit",
                   choices=["vit", "vit_convwin", "resnet18"],
                   help="choose baseline vit, improved vit (conv+window), or resnet18")
    return p.parse_args()



@dataclass
class Cfg:
    data_root: str
    batch_size: int
    num_workers: int
    # 训练超参
    epochs: int = 100
    lr: float = 3e-4                   # warmup+cosine 的 base lr
    weight_decay: float = 0.05
    warmup_ratio: float = 0.05         # 5% steps 用于 warmup
    min_lr: float = 1e-6               # 余弦衰减到的最小 lr
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# 你可以改这里的默认 batch/epochs
cfg = Cfg(
    data_root="./data",
    batch_size=512,
    num_workers=8,
    epochs=50,
)

CIFAR_MEAN = (0.4914, 0.4822, 0.4465) 
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# =========================
# 2) Fix Seed
# =========================
def set_seed(seed: int = 42, deterministic: bool = True):
    """
    固定全链路随机性：Python、NumPy、PyTorch (CPU/GPU)、DataLoader、cuDNN。
    注意：deterministic=True 会牺牲一点性能，换取更强可复现性。
    """
    # 1) Python & NumPy & PyTorch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2) cuDNN & 算法确定性
    # 关闭 benchmark，避免根据输入数据动态选择非确定性算法
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 3) 强制使用确定性算子（某些算子可能报错，需要你替换成确定性实现）
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    # 4) 某些 CUDA 内核在完全确定性时需要这个环境变量（PyTorch 官方建议）
    # 仅在 Linux + CUDA 场景有用，无害
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def seed_worker(worker_id):
    # 每个 worker 都基于主进程 seed 派生一个确定的子种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



# =========================
# 3) 可视化（含反归一化）
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
# 4) 训练 & 验证
# =========================
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, lr_history, print_every=100):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)
    step_in_epoch = 0

    for imgs, labels in pbar:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # ---- scheduler: 每个 step 更新 ----
        scheduler.step()

        # 记录 & 偶尔打印学习率
        curr_lr = optimizer.param_groups[0]['lr']
        lr_history.append(curr_lr)
        step_in_epoch += 1
        if step_in_epoch % print_every == 0:
            print(f"  [step {step_in_epoch:4d}] lr={curr_lr:.6e}")

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
# 5) 画曲线（含 LR 曲线 + 模型名）
# =========================
def plot_curves(history, batch_size, epochs, model_name, outdir="checkpoints"):
    os.makedirs(outdir, exist_ok=True)
    x_epoch = range(1, len(history["train_loss"]) + 1)
    x_step = range(1, len(history["lr"]) + 1)

    # Loss
    plt.figure()
    plt.plot(x_epoch, history["train_loss"], label="train")
    plt.plot(x_epoch, history["val_loss"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Loss over epochs — {model_name}")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"loss_curve_{model_name}_bs{batch_size}_ep{epochs}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(x_epoch, history["train_acc"], label="train")
    plt.plot(x_epoch, history["val_acc"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"Accuracy over epochs — {model_name}")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"acc_curve_{model_name}_bs{batch_size}_ep{epochs}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

    # LR（按 step）
    plt.figure()
    plt.plot(x_step, history["lr"], label="lr")
    plt.xlabel("Step"); plt.ylabel("Learning Rate")
    plt.title(f"LR over steps (warmup+cosine) — {model_name}")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"lr_curve_{model_name}_bs{batch_size}_ep{epochs}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


# =========================
# 6) 主程序
# =========================
def main(cfg: Cfg):
    args = parse_args()
    set_seed(42, deterministic=True)
    device = torch.device(cfg.device)
    print(f"Using device: {device} | model={args.model}")

    train_loader, val_loader = get_dataloaders(cfg)

    if args.model == "vit":
        model = build_model("vit")                # 默认超参：config.VIT
        model_name = "vit"
    elif args.model == "vit_convwin":
        model = build_model("vit_convwin")        # 默认超参：config.VIT_CONVWIN
        model_name = "vit_convwin"
    else:  # resnet18
        # CIFAR-10 上最合理的默认：small_input=True, 无预训练
        model = build_model("resnet18", num_classes=10, small_input=True)
        model_name = "resnet18"

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- warmup+cosine (step) 与 ViT 共用 ----
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    base_lr, min_lr = cfg.lr, cfg.min_lr

    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cosine
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    for epoch in range(cfg.epochs):
        print(f"\nEpoch [{epoch+1}/{cfg.epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, history["lr"])
        print(f" [epoch {epoch+1:3d}] lr={optimizer.param_groups[0]['lr']:.6e}")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{model_name}_cifar10_best.pth")
            print(f" ✓ Saved best model (acc={best_acc:.4f})")

    plot_curves(history, cfg.batch_size, cfg.epochs, model_name, outdir="checkpoints")
    print("\nDone.")

if __name__ == "__main__":
    main(cfg)