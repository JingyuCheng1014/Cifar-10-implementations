import os
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from data.cifar10 import get_dataloaders
from model.resnet18 import resnet18
from config import cfg as cfg_from_config, VIT, build_model
from model.vit import ViT
from model.swin import swin_t, swin_s, swin_b, swin_l




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
# 3) 提取特征并做PCA
# =========================
def extract_features(model, loader, device, feature_layer=None):
    model.eval()
    features = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extract", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            # 针对不同模型结构，获取倒数第二层特征
            if hasattr(model, "forward_features"):
                feats = model.forward_features(imgs)
            elif hasattr(model, "get_intermediate"):
                feats = model.get_intermediate(imgs)
            elif isinstance(model, ViT):
                # For ViT, extract patch embeddings before classification head
                x = model.to_patch_embedding(imgs)
                b, n, _ = x.shape
                cls_tokens = model.cls_token.expand(b, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x += model.pos_embedding[:, :(n + 1)]
                x = model.dropout(x)
                x = model.transformer(x)
                # Return CLS token for consistent comparison with ResNet features
                feats = x[:, 0]  # (B, dim)
            else:
                # 通用做法：去掉最后一层全连接
                feats = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(imgs))))))))
                feats = feats.view(feats.size(0), -1)
            features.append(feats.cpu())
            labels_list.append(labels.cpu())
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features.numpy(), labels.numpy()


def plot_pca(features, labels, classes, model_name, outdir="PCA"):
    os.makedirs(outdir, exist_ok=True)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(8,6))
    for i, c in enumerate(classes):
        idx = labels == i
        plt.scatter(reduced[idx,0], reduced[idx,1], label=c, s=10, alpha=0.6)
    plt.legend(markerscale=2, fontsize=10)
    plt.title(f"PCA of {model_name} features")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_pca.png"), dpi=150)
    plt.close()
    print(f"PCA plot saved: {os.path.join(outdir, f'{model_name}_pca.png')}")
    
def plot_pca_comparison(resnet_features, vit_features, labels, classes, outdir="PCA"):
    """
    Compare PCA plots of ResNet and ViT features side by side
    """
    os.makedirs(outdir, exist_ok=True)
    
    # ResNet PCA
    resnet_pca = PCA(n_components=2)
    resnet_reduced = resnet_pca.fit_transform(resnet_features)
    
    # ViT PCA
    vit_pca = PCA(n_components=2)
    vit_reduced = vit_pca.fit_transform(vit_features)
    
    # Plot side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ResNet plot
    for i, c in enumerate(classes):
        idx = labels == i
        ax1.scatter(resnet_reduced[idx,0], resnet_reduced[idx,1], label=c, s=10, alpha=0.6)
    ax1.set_title(f"ResNet18 Features PCA")
    ax1.legend(markerscale=2, fontsize=8)
    
    # ViT plot
    for i, c in enumerate(classes):
        idx = labels == i
        ax2.scatter(vit_reduced[idx,0], vit_reduced[idx,1], label=c, s=10, alpha=0.6)
    ax2.set_title(f"ViT Features PCA")
    ax2.legend(markerscale=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "resnet_vs_vit_pca_comparison.png"), dpi=150)
    plt.close()
    print(f"PCA comparison plot saved: {os.path.join(outdir, 'resnet_vs_vit_pca_comparison.png')}")


# =========================
# 4) 主流程
# =========================
def run_and_pca(model, model_name, train_loader, val_loader, classes, device, cfg):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    base_lr = cfg.lr
    min_lr = cfg.min_lr

    from torch.optim.lr_scheduler import LambdaLR
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    print(f"\n==== Training {model_name} ====")
    t0 = time.perf_counter()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc = 0.0
    best_state = None
    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device, history["lr"])
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(best_state, f"checkpoints/{model_name}_best.pth")
            print(f" ✓ Saved best model (acc={best_acc:.4f})")
    t1 = time.perf_counter()
    sec_total = t1 - t0
    sec_per_epoch = sec_total / cfg.epochs
    print(f"{model_name} finished. Best Val Acc: {best_acc:.4f} | sec/epoch: {sec_per_epoch:.3f} | total: {sec_total:.1f}s")

    # 保存训练日志
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", f"pca_{model_name}_log.csv")
    with open(log_path, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(cfg.epochs):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i]
            ])
    print(f"Log saved: {log_path}")

    # PCA on validation set
    features, labels = extract_features(model, val_loader, device)
    plot_pca(features, labels, classes, model_name, outdir="PCA")
    return best_acc, sec_per_epoch



def main(cfg: Cfg):
    device = torch.device(cfg.device)
    train_loader, val_loader = get_dataloaders(cfg)
    classes = train_loader.dataset.classes

    # Store features and results for all models
    all_features = {}
    all_labels = None

    # ViT
    model_vit = build_model("vit", **VIT).to(device)
    vit_acc, vit_sec = run_and_pca(model_vit, "vit", train_loader, val_loader, classes, device, cfg)
    vit_features, labels = extract_features(model_vit, val_loader, device)
    all_features["vit"] = vit_features
    all_labels = labels

    # ResNet18
    model_resnet = resnet18(num_classes=10).to(device)
    resnet_acc, resnet_sec = run_and_pca(model_resnet, "resnet18", train_loader, val_loader, classes, device, cfg)
    resnet_features, _ = extract_features(model_resnet, val_loader, device)
    all_features["resnet18"] = resnet_features

    # Swin Transformer
    model_swin = swin_t(num_classes=10).to(device)
    swin_acc, swin_sec = run_and_pca(model_swin, "swin_t", train_loader, val_loader, classes, device, cfg)
    swin_features, _ = extract_features(model_swin, val_loader, device)
    all_features["swin_t"] = swin_features

    # Generate individual PCA plots
    print("\n==== Generating individual PCA plots ====")
    for model_name, features in all_features.items():
        plot_pca(features, all_labels, classes, model_name, outdir="PCA")

    # Generate comparison plots
    print("\n==== Generating comparison plots ====")
    # ResNet vs ViT
    plot_pca_comparison(all_features["resnet18"], all_features["vit"], all_labels, classes, outdir="PCA")
    
    # Create a function to plot all three models side by side
    def plot_pca_comparison_three(resnet_features, vit_features, swin_features, labels, classes, outdir="PCA"):
        """
        Compare PCA plots of ResNet, ViT and Swin features side by side
        """
        os.makedirs(outdir, exist_ok=True)
        
        # ResNet PCA
        resnet_pca = PCA(n_components=2)
        resnet_reduced = resnet_pca.fit_transform(resnet_features)
        
        # ViT PCA
        vit_pca = PCA(n_components=2)
        vit_reduced = vit_pca.fit_transform(vit_features)
        
        # Swin PCA
        swin_pca = PCA(n_components=2)
        swin_reduced = swin_pca.fit_transform(swin_features)
        
        # Plot side by side
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        # ResNet plot
        for i, c in enumerate(classes):
            idx = labels == i
            ax1.scatter(resnet_reduced[idx,0], resnet_reduced[idx,1], label=c, s=10, alpha=0.6)
        ax1.set_title(f"ResNet18 Features PCA")
        ax1.legend(markerscale=2, fontsize=8)
        
        # ViT plot
        for i, c in enumerate(classes):
            idx = labels == i
            ax2.scatter(vit_reduced[idx,0], vit_reduced[idx,1], label=c, s=10, alpha=0.6)
        ax2.set_title(f"ViT Features PCA")
        ax2.legend(markerscale=2, fontsize=8)
        
        # Swin plot
        for i, c in enumerate(classes):
            idx = labels == i
            ax3.scatter(swin_reduced[idx,0], swin_reduced[idx,1], label=c, s=10, alpha=0.6)
        ax3.set_title(f"Swin Transformer Features PCA")
        ax3.legend(markerscale=2, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "resnet_vit_swin_pca_comparison.png"), dpi=150)
        plt.close()
        print(f"PCA three-way comparison plot saved: {os.path.join(outdir, 'resnet_vit_swin_pca_comparison.png')}")

    # Generate three-way comparison
    plot_pca_comparison_three(
        all_features["resnet18"], 
        all_features["vit"], 
        all_features["swin_t"], 
        all_labels, 
        classes, 
        outdir="PCA"
    )

    print("\n=== Summary ===")
    print(f"ResNet18: acc={resnet_acc:.4f}, sec/epoch={resnet_sec:.3f}")
    print(f"ViT     : acc={vit_acc:.4f}, sec/epoch={vit_sec:.3f}")
    print(f"Swin T  : acc={swin_acc:.4f}, sec/epoch={swin_sec:.3f}")

if __name__ == "__main__":
    main(cfg)
