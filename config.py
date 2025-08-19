# config.py
from dataclasses import dataclass
import torch

# ============== 基础训练配置（与 train.py 中一致） ==============
@dataclass
class Cfg:
    # 数据
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4

    # 训练
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.05         # ★ 新增：与 train.py 对齐
    warmup_ratio: float = 0.05         # ★ 新增
    min_lr: float = 1e-6               # ★ 新增
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 8539

    # 路径
    checkpoints_dir: str = "./checkpoints"

# 一个全局实例，train.py 可直接 `from config import cfg`
cfg = Cfg()


# ============== 模型超参（ViT - 基线） ==============
# 与 model/vit.py 的参数名完全对齐
VIT = dict(
    image_size=32,   # CIFAR-10
    patch_size=4,    # 32 可被 4 整除 => 8x8=64 patches
    num_classes=10,
    dim=256,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1,
)

# ============== 模型超参（ViTConvWin - 改进版） ==============
# 与 model/vit.py 中 ViTConvWin 的参数名对齐
VIT_CONVWIN = dict(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=256,
    depth=6,
    heads=8,
    mlp_ratio=4.0,
    window_size=4,     # 小图像 32x32 时推荐 4
    in_chans=3,
    dropout=0.0,
    attn_drop=0.0,
)


# ============== 简单模型工厂（统一入口） ==============
# config.py 中
def build_model(name: str = "vit", **overrides):
    name = name.lower()
    if name == "vit":
        from model.vit import ViT
        params = {**VIT, **overrides}
        return ViT(**params)
    elif name == "vit_convwin":
        from model.vit import ViTConvWin
        params = {**VIT_CONVWIN, **overrides}
        return ViTConvWin(**params)
    elif name == "resnet18":
        from model.resnet18 import resnet18
        num_classes = overrides.get("num_classes", VIT["num_classes"])
        small_input = overrides.get("small_input", True)
        weights = overrides.get("weights", None)  # e.g. ResNet18_Weights.DEFAULT
        return resnet18(num_classes=num_classes, small_input=small_input, weights=weights)
    else:
        raise ValueError(f"Unknown model name: {name}")

