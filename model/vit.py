import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def pair(t):
    """
    将输入参数转化为元组。
    Args:
        t (any): 输入参数，可以是任意类型。

    Returns:
        tuple: 如果输入参数已经是元组，则返回原元组；否则返回一个新元组，元组中的两个元素均为输入参数的值。
    """
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, dim, **kwargs):
        """
        把图像切成不重叠patch并映射到dim维度的embedding空间
        输入:  (B, C, H, W)
        输出:  (B, N, dim)  其中 N = (H/Ph) * (W/Pw)
        """
        super().__init__()
        self.num_channels = kwargs.get('num_channels', 3)

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by patch size'

        self.patch_height = patch_height
        self.patch_width = patch_width

        # patch 数和每个 patch 的原始维度
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = self.num_channels * patch_height * patch_width

        # 线性投影到 dim
        self.proj = nn.Linear(patch_dim, dim)

        # （可选）做个轻量归一化，稳定训练
        self.pre_norm = nn.LayerNorm(patch_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        return: (B, N, dim)
        """
        # 把每个 patch 展平成一个向量: (B, C, H, W) -> (B, N, Ph*Pw*C)
        x = rearrange(
            x,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=self.patch_height, p2=self.patch_width
        )
        x = self.pre_norm(x)
        x = self.proj(x)  # (B, N, dim)
        return x


class Embeddings(nn.Module):
    def __init__(self, image_size, patch_size, dim, **kwargs):
        """
        组合 PatchEmbedding + [CLS] token + 位置编码 + dropout
        最终输出给 Transformer 的序列: (B, N+1, dim)
        """
        super().__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, dim, **kwargs)

        # 可学习的 [CLS] token，shape = (1, 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 可学习的位置编码，包含 N 个 patch 再加 1 个 CLS 位置
        num_patches = self.patch_embedding.num_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # dropout
        self.dropout = nn.Dropout(kwargs.get('emb_dropout', 0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) -> (B, N+1, dim)
        """
        # 1) patch embed
        x = self.patch_embedding(x)        # (B, N, dim)
        B, N, D = x.shape

        # 2) prepend CLS token
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)  # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)                       # (B, N+1, dim)

        # 3) add positional embedding（按实际长度截取）
        x = x + self.pos_embedding[:, :N+1, :]

        # 4) dropout
        x = self.dropout(x)
        return x




class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        初始化多头注意力机制模块。
        Args:
            dim (int): 输入特征的维度。
            heads (int, optional): 多头注意力机制的头数，默认为8。
            dim_head (int, optional): 每个头的维度，默认为64。
            dropout (float, optional): Dropout的比率，默认为0。
        Returns:
            None
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        对输入x进行前向传播，计算并返回注意力机制的输出。
        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_length, embedding_dim)。
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_size, seq_length, embedding_dim)。
        """
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)






class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        初始化函数。
        Args:
            dim (int): 输入和输出的维度。
            hidden_dim (int): 隐藏层的维度。
            dropout (float, optional): Dropout比率，用于防止过拟合。默认为0.0。
        Returns:
            None
        Raises:
            无
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)





class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        Transformer 模型的初始化函数。
        Args:
            dim (int): 输入和输出的维度。
            depth (int): Transformer 的层数。
            heads (int): 多头注意力机制中头的数量。
            dim_head (int): 每个头的维度。
            mlp_dim (int): 前馈网络的维度。
            dropout (float, optional): Dropout 的比例，默认为 0.0。
        Returns:
            None
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)




class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        """
        初始化Vision Transformer模型。
        Args:
            image_size (tuple): 输入图像的尺寸，格式为(高度, 宽度)。
            patch_size (tuple): 补丁的尺寸，格式为(高度, 宽度)。
            num_classes (int): 类别数。
            dim (int): Transformer模型中隐藏层的维度。
            depth (int): Transformer模型的层数。
            heads (int): 多头注意力机制中的头数。
            mlp_dim (int): 多层感知机中的隐藏层维度。
            pool (str): 池化类型，可选值为'cls'（类标记池化）或'mean'（平均池化）。
            channels (int): 输入图像的通道数，默认为3（RGB图像）。
            dim_head (int): 多头注意力机制中每个头的维度，默认为64。
            dropout (float): Transformer模型中的丢弃率，默认为0。
            emb_dropout (float): 嵌入层中的丢弃率，默认为0。
        Returns:
            None
        Raises:
            AssertionError: 如果图像尺寸不能被补丁尺寸整除，或者池化类型不是'cls'或'mean'，将引发断言错误。
        """
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        dim_head = dim // heads

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


