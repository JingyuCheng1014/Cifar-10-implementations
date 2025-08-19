# train_ablation.py
import os, time, csv
import random, numpy as np
from dataclasses import dataclass
import argparse
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# === 与 train.py 对齐：数据与模型统一从项目模块入口拿 ===
from data.cifar10 import get_dataloaders
from config import cfg as cfg_from_config, VIT, VIT_CONVWIN, build_model


# =========================
# 1) 命令行参数：与 train.py 一致
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="vit",
                   choices=["vit", "vit_convwin"],
                   help="choose baseline vit or improved vit with conv+window")
    return p.parse_args()


# =========================
# 2) 训练配置（数值与 train.py 对齐）
# =========================
@dataclass
class Cfg:
    data_root: str
    batch_size: int
    num_workers: int
    # 训练超参
    epochs: int = 50
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
    epochs=50
)

# 可视化时用到
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)


# =========================
# 3) 与 train.py 一致的随机性固定
# =========================
def set_seed(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# =========================
# 4) 训练 & 验证（完全复用 train.py 的实现风格）
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
# 5) 单次配置训练（核心：通过 build_model + overrides）
# =========================
def run_ablation(
    train_loader, val_loader, device, cfg, args,
    patch_size=4, dim=256, depth=6, heads=8, tag_prefix=""
):
    """
    训练一个配置，返回 best_val_acc, sec/epoch, params，并将 best checkpoint 与曲线保存到磁盘。
    与 train.py 一致地：optimizer / scheduler / loss / log。
    模型构造通过 config.build_model(args.model, **overrides)。
    """
    set_seed(42)

    assert dim % heads == 0, f"d_model {dim} must be divisible by nhead {heads}"

    # 配置名（带模型前缀，避免与基线/改进版重名）
    model_name = args.model
    config_name = f"{tag_prefix}{model_name}_ps{patch_size}_dim{dim}_d{depth}_h{heads}"
    print(f"\n[Run] {config_name}")

    # 仅把需要消融的参数作为 overrides 传入，默认值由 VIT / VIT_CONVWIN 提供
    overrides = dict(
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
    )
    # 提醒：如果你想对 vit_convwin 额外固定 window_size=4，也可以在 overrides 里加：
    # if args.model == "vit_convwin":
    #     overrides["window_size"] = 4  # 例子：与 train.py 的默认一致

    # ⚠️ 保留：若要直接 import ViT 的写法（不建议）：
    # from model.vit import ViT
    model = build_model(args.model, **overrides).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ==== warmup + cosine scheduler（按 step 更新）====
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

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc, best_state = 0.0, None

    t0 = time.perf_counter()
    for epoch in range(cfg.epochs):
        print(f"\nEpoch [{epoch+1}/{cfg.epochs}]")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, history["lr"])
        print(f" [epoch {epoch+1:3d}] lr={optimizer.param_groups[0]['lr']:.6e}")
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"BestVal={best_val_acc:.4f}")

    sec_per_epoch = (time.perf_counter() - t0) / cfg.epochs
    params = sum(p.numel() for p in model.parameters())

    # 保存 best ckpt 与曲线
    torch.save(best_state, f"checkpoints/{config_name}_best.pth")
    with open(f"logs/{config_name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(cfg.epochs):
            w.writerow([i+1, history["train_loss"][i], history["train_acc"][i], history["val_loss"][i], history["val_acc"][i]])

    print(f"  ✓ Saved: checkpoints/{config_name}_best.pth | logs/{config_name}.csv")
    print(f"  ✓ Params: {params / 1e6:.2f}M | sec/epoch: {sec_per_epoch:.3f}")
    return best_val_acc, sec_per_epoch, params, history, config_name


# =========================
# 6) 画曲线
# =========================
def plot_curves(history, batch_size, epochs, model_name, outdir="checkpoints_ablation/plot"):
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
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(x_epoch, history["train_acc"], label="train")
    plt.plot(x_epoch, history["val_acc"], label="val")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title(f"Accuracy over epochs — {model_name}")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"acc_curve_{model_name}_bs{batch_size}_ep{epochs}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # LR（按 step）
    plt.figure()
    plt.plot(x_step, history["lr"], label="lr")
    plt.xlabel("Step"); plt.ylabel("Learning Rate")
    plt.title(f"LR over steps (warmup+cosine) — {model_name}")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"lr_curve_{model_name}_bs{batch_size}_ep{epochs}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# 7) 主流程：与 train.py 同结构 + 消融循环
# =========================
def main(cfg: Cfg):
    args = parse_args()
    set_seed(42)
    device = torch.device(cfg.device)
    print(f"Using device: {device} | model={args.model}")

    train_loader, val_loader = get_dataloaders(cfg)

    # === 设定基线（与 train.py 的默认一致）
    if args.model == "vit":
        base_cfg = dict(patch_size=4, dim=256, depth=6, heads=8)
    else:  # vit_convwin
        # 与 VIT_CONVWIN 的默认超参保持一致；消融主轴沿用 vit 的四个
        base_cfg = dict(patch_size=4, dim=256, depth=6, heads=8)

    print("\n=== Baseline ===")
    base_acc, base_sec, base_params, base_history, base_name = run_ablation(
        train_loader, val_loader, device, cfg, args, **base_cfg, tag_prefix="base_")
    print(f"Baseline Acc={base_acc:.4f}, Params={base_params/1e6:.2f}M, sec/epoch={base_sec:.3f}")
    plot_curves(base_history, cfg.batch_size, cfg.epochs, base_name)

    results = []

    # Patch Size Ablation
    print("\n=== Patch Size Ablation ===")
    best_patch = None
    patch_results = []
    for patch in [2, 4, 8]:
        acc, sec, params, history, name = run_ablation(
            train_loader, val_loader, device, cfg, args,
            patch_size=patch, dim=base_cfg["dim"], depth=base_cfg["depth"],
            heads=base_cfg["heads"], tag_prefix="patch_")
        results.append(("patch", patch, acc, sec, params))
        patch_results.append((patch, params, sec, acc))
        plot_curves(history, cfg.batch_size, cfg.epochs, name)
        if (best_patch is None) or (acc > best_patch[1]):
            best_patch = (patch, acc)
    with open("logs/ps.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ps", "Params", "sec/epoch", "accuracy"])
        for row in patch_results: w.writerow(row)

    # Embedding Dim Ablation
    print("\n=== Embedding Dimension Ablation ===")
    best_dim = None
    dim_results = []
    for dim in [96, 192, 256]:
        heads = base_cfg["heads"]
        if dim % heads != 0:
            continue
        acc, sec, params, history, name = run_ablation(
            train_loader, val_loader, device, cfg, args,
            patch_size=base_cfg["patch_size"], dim=dim, depth=base_cfg["depth"],
            heads=heads, tag_prefix="dim_")
        results.append(("dim", dim, acc, sec, params))
        dim_results.append((dim, params, sec, acc))
        plot_curves(history, cfg.batch_size, cfg.epochs, name)
        if (best_dim is None) or (acc > best_dim[1]):
            best_dim = (dim, acc)
    with open("logs/dim.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dim", "Params", "sec/epoch", "accuracy"])
        for row in dim_results: w.writerow(row)

    # Depth Ablation
    print("\n=== Depth Ablation ===")
    best_depth = None
    depth_results = []
    for depth in [4, 6, 12]:
        acc, sec, params, history, name = run_ablation(
            train_loader, val_loader, device, cfg, args,
            patch_size=base_cfg["patch_size"], dim=base_cfg["dim"],
            depth=depth, heads=base_cfg["heads"], tag_prefix="depth_")
        results.append(("depth", depth, acc, sec, params))
        depth_results.append((depth, params, sec, acc))
        plot_curves(history, cfg.batch_size, cfg.epochs, name)
        if (best_depth is None) or (acc > best_depth[1]):
            best_depth = (depth, acc)
    with open("logs/depth.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["depth", "Params", "sec/epoch", "accuracy"])
        for row in depth_results: w.writerow(row)

    # Heads Ablation
    print("\n=== Heads Ablation ===")
    best_heads = None
    heads_results = []
    for heads in [1, 2, 4, 8, 128]:
        dim = base_cfg["dim"]
        if dim % heads != 0:
            continue
        acc, sec, params, history, name = run_ablation(
            train_loader, val_loader, device, cfg, args,
            patch_size=base_cfg["patch_size"], dim=dim,
            depth=base_cfg["depth"], heads=heads, tag_prefix="heads_")
        results.append(("heads", heads, acc, sec, params))
        heads_results.append((heads, params, sec, acc))
        plot_curves(history, cfg.batch_size, cfg.epochs, name)
        if (best_heads is None) or (acc > best_heads[1]):
            best_heads = (heads, acc)
    with open("logs/heads.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["heads", "Params", "sec/epoch", "accuracy"])
        for row in heads_results: w.writerow(row)

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
    final_acc, final_sec, final_params, final_history, final_name = run_ablation(
        train_loader, val_loader, device, cfg, args, **final_cfg, tag_prefix="final_")
    plot_curves(final_history, cfg.batch_size, cfg.epochs, final_name)
    print(f"\nFinal Combo: {final_cfg} | acc={final_acc:.4f} | params={final_params / 1e6:.2f}M | sec/epoch={final_sec:.3f}")

    # 写入best_csv
    with open("logs/best_csv.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "Params", "sec/epoch", "accuracy"])
        w.writerow(["", final_params, final_sec, final_acc])

    print("\nDone.")


if __name__ == "__main__":
    main(cfg)
