import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def get_dataloaders(cfg):
    # 定义数据增强流程
    transform = T.Compose([
        # 调整图像大小为cfg.image_size
        T.Resize(cfg.image_size),
        # 将图像转换为张量
        T.ToTensor(),
        # 对图像进行归一化处理
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建训练数据集
    train_ds = CIFAR10(cfg.data_root, train=True, download=True, transform=transform)
    # 创建验证数据集
    val_ds = CIFAR10(cfg.data_root, train=False, download=True, transform=transform)

    # 创建训练数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 创建验证数据加载器
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    # 返回训练数据加载器和验证数据加载器
    return train_loader, val_loader
