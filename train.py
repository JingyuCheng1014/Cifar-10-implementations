import os
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.cifar10 import get_dataloaders
# ====== 关键：引入你给的 Swin 版本（swin_t/swin_s/...）======
from model.swin import swin_t, swin_s, swin_b, swin_l
# from config import cfg as cfg_from_config, VIT, build_model



# =========================
# 1) 配置
# =========================
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
    num_workers=4,
    epochs=100,
)


# =========================
# 2) 仅用于可视化的均值 / 方差
# =========================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


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
    import time

    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # 数据
    train_loader, val_loader = get_dataloaders(cfg)

    # ====== 构建 Swin 模型（CIFAR-10: 32x32）======
    # 注意：downscaling_factors 默认为 (4,2,2,2)，乘积为 32，正好匹配 32x32
    model = swin_t(
        hidden_dim=96,
        layers=(2, 2, 6, 2),
        heads=(3, 6, 12, 24),
        head_dim=32,
        window_size=4,   # 小图像建议 4
        channels=3,
        num_classes=10,
        downscaling_factors=(4, 2, 1, 1),
        relative_pos_embedding=True,
    ).to(device)
    model_name = "swin_t"

    # ==== 如需改成 vit，只需替换 model 并改 model_name ====
    # from model.vit import ViT
    # model_name = "vit"
    # model = build_model("vit", **VIT).to(device)

    # 损失函数（可加 label smoothing）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ---- warmup + cosine scheduler（按 step 更新）----
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    base_lr = cfg.lr
    min_lr = cfg.min_lr

    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))  # 0->1
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1->0
        return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 历史记录
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    # 统计参数量
    params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {params / 1e6:.2f}M")

    # 训练
    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        print(f"\nEpoch [{epoch+1}/{cfg.epochs}]")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, history["lr"]
        )

        # 每个 epoch 末尾打印一次当前 lr
        print(f" [epoch {epoch+1:3d}] lr={optimizer.param_groups[0]['lr']:.6e}")

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f}")

        # 记录
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/{model_name}_cifar10_best.pth")
            print(f" ✓ Saved best model (acc={best_acc:.4f})")

    sec_per_epoch = (time.perf_counter() - t0) / cfg.epochs
    print(f"\nTraining finished. sec/epoch: {sec_per_epoch:.3f}")

    # === 训练完成后：画曲线并保存 ===
    plot_curves(history, cfg.batch_size, cfg.epochs, model_name, outdir="checkpoints")
    print("\nDone.")


if __name__ == "__main__":
    main(cfg)
