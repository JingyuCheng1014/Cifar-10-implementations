import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops import rearrange, repeat



# =========================
# 1) Baseline ViT
# =========================

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





# =========================
# 2) Improved ViT(Conv Patch + Window MSA)
# =========================



# ---------- 1) 卷积式 Patch Embedding（Swin 思路） ----------
class ConvPatchEmbed(nn.Module):
    """
    Conv2d(kernel=stride=patch) 做 patch 投影；可选 token 级 LayerNorm
    输出:
      tokens: (B, N, C), grid: (H, W)
    """
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        # B,C,H,W -> B,embed_dim,H',W'
        x = self.proj(x)
        B, C, H, W = x.shape
        # B,N,C
        x = x.flatten(2).transpose(1, 2).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x, (H, W)

# ---------- 2) window 工具（Swin 的 window process 思路） ----------
def window_partition(x, window_size):
    """
    x: (B, H, W, C) -> (num_windows*B, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0, "H/W must be divisible by window_size"
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # (B, nH, nW, ws, ws, C) -> (B*nH*nW, ws*ws, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    windows: (num_windows*B, ws*ws, C) -> x: (B, H, W, C)
    """
    B = int(windows.shape[0] // (H * W // window_size // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

# ---------- 3) 相对位置偏置（简版，参考 Swin） ----------
def get_rel_pos_index(window_size):
    """
    生成窗口内 pair-wise 相对坐标索引表，用于查 relative_position_bias_table
    返回 (ws*ws, ws*ws) 的 index 矩阵，值域 [0, (2*ws-1)^2)
    """
    ws = window_size
    coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij'))  # 2, ws, ws
    coords_flatten = torch.flatten(coords, 1)  # 2, ws*ws
    rel_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
    rel_coords = rel_coords.permute(1, 2, 0).contiguous()  # N, N, 2
    rel_coords[:, :, 0] += ws - 1
    rel_coords[:, :, 1] += ws - 1
    rel_coords[:, :, 0] *= 2 * ws - 1
    rel_pos_index = rel_coords.sum(-1)  # N, N
    return rel_pos_index  # (ws*ws, ws*ws)

class WindowMSA(nn.Module):
    """
    窗口内多头注意力（可选相对位置偏置）；输入 shape: (num_windows*B, N, C)
    """
    def __init__(self, dim, window_size=4, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., use_rel_pos_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_rel_pos_bias = use_rel_pos_bias
        if use_rel_pos_bias:
            # (2*ws-1)^2, nH
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            rel_index = get_rel_pos_index(window_size)  # (N, N)
            self.register_buffer("relative_position_index", rel_index, persistent=False)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)  # each: (B_, N, nH, d)
        q = q.permute(0,2,1,3)  # B_, nH, N, d
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B_, nH, N, N

        if self.use_rel_pos_bias:
            rel_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            rel_bias = rel_bias.view(self.window_size * self.window_size,
                                     self.window_size * self.window_size, -1)  # N,N,nH
            rel_bias = rel_bias.permute(2,0,1).unsqueeze(0)  # 1,nH,N,N
            attn = attn + rel_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class WinBlock(nn.Module):
    """
    单个 block：LN -> WindowMSA -> 残差；LN -> MLP -> 残差
    这里不实现 Swin 的 shift（保持简单稳定），如需 SW-MSA 可再加 shift+mask。
    """
    def __init__(self, dim, window_size=4, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowMSA(dim, window_size, num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.window_size = window_size

    def forward(self, x, H, W):
        # x: (B, N, C)
        B, N, C = x.shape
        ws = self.window_size
        # (B,N,C)->(B,H,W,C)
        x_ = x.view(B, H, W, C)

        # window attention
        windows = window_partition(self.norm1(x_), ws)                      # (BnW, ws*ws, C)
        windows = self.attn(windows)                                        # (BnW, ws*ws, C)
        x_ = window_reverse(windows, ws, H, W)                              # (B,H,W,C)

        x = x + x_.view(B, N, C)                                           # 残差
        x = x + self.mlp(self.norm2(x))                                    # FFN 残差
        return x

# ---------- 4) 改进版 ViT（卷积 patch + window MSA） ----------
class ViTConvWin(nn.Module):
    """
    改进版 ViT：
      - Conv patch embedding（Swin风格）
      - 每层使用窗口化 self-attention（非 shift），计算复杂度降到 O(nW * ws^4)
      - 末端用 mean pooling（不使用 CLS token）
    """
    def __init__(self, *, image_size=32, patch_size=4, num_classes=10,
                 dim=256, depth=6, heads=8, mlp_ratio=4., window_size=4,
                 in_chans=3, dropout=0., attn_drop=0.):
        super().__init__()
        self.patch_embed = ConvPatchEmbed(img_size=image_size, patch_size=patch_size,
                                          in_chans=in_chans, embed_dim=dim, norm_layer=nn.LayerNorm)
        self.blocks = nn.ModuleList([
            WinBlock(dim=dim, window_size=window_size, num_heads=heads,
                     mlp_ratio=mlp_ratio, drop=dropout, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, img):
        # tokens: (B,N,C), grid: (H',W')
        x, (H, W) = self.patch_embed(img)
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.norm(x)
        # mean pooling（Swin 不用 CLS）
        x = x.mean(dim=1)
        return self.head(x)