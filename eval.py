# eval.py
# 用法：
#   python eval.py --ckpt checkpoints/tinycnn_cifar10_best.pth
#   可选参数：--data_root ./data --batch_size 128 --num_workers 4 --device auto

import os
import argparse
import torch
import torch.nn as nn

# 直接复用 train.py 里已实现的组件（不会触发训练，因为有 if __name__ == "__main__": 保护）
from train import TinyCNN, Cfg, get_dataloaders, evaluate as eval_fn

def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    """加载权重，兼容 DataParallel 保存的 'module.' 前缀。"""
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # 处理 DataParallel 前缀
    new_state = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        new_state[k] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 classifier")
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint .pth')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args()

    # device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"[Info] Using device: {device}")

    # cfg（只用到 data_root / batch_size / num_workers）
    cfg = Cfg(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # dataloader（只取验证集）
    _, val_loader = get_dataloaders(cfg)

    # model & loss
    model = TinyCNN(num_classes=10).to(device)
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    load_checkpoint(model, args.ckpt)
    criterion = nn.CrossEntropyLoss()

    # eval
    val_loss, val_acc = eval_fn(model, val_loader, criterion, device)
    print(f"\n[Eval] loss={val_loss:.4f}, acc={val_acc:.4f}")

if __name__ == '__main__':
    main()
