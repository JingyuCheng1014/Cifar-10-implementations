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
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 8539

    # 路径
    checkpoints_dir: str = "./checkpoints"

# 一个全局实例，train.py 可直接 `from config import cfg`
cfg = Cfg()


# ============== 模型超参（ViT） ==============
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


# ============== 简单模型工厂（可选，用于统一入口） ==============
def build_model(name: str = "vit", **overrides):
    """
    用法：
      from config import build_model, VIT
      model = build_model("vit", **VIT)               # 直接用默认
      model = build_model("vit", **{**VIT, "dim":384})# 覆盖部分超参
    """
    name = name.lower()
    if name == "vit":
        from model.vit import ViT
        params = {**VIT, **overrides}
        return ViT(**params)

    elif name == "resnet18":
        # 需要你在 model/resnet18.py 暴露 resnet18(num_classes=...)
        from model.resnet18 import resnet18
        num_classes = overrides.get("num_classes", VIT["num_classes"])
        return resnet18(num_classes=num_classes)

    elif name == "swin":
        # 如果后续要接入 Swin，可在此补充默认超参
        from model.swin import Swin
        return Swin(**overrides)

    else:
        raise ValueError(f"Unknown model name: {name}")
