import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
try:
    from timm.layers import DropPath, to_2tuple
except ImportError:
    from timm.models.layers import DropPath, to_2tuple

try:
    from flash_attn.modules.mha import FlashSelfAttention
except ImportError:
    FlashSelfAttention = None

from .attention import SD_attn
from .lora import LoRAMode, LoRARollout


def get_pad3d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon)

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back


def get_pad2d(input_resolution, window_size):
    """
    Args:
        input_resolution (tuple[int]): Lat, Lon
        window_size (tuple[int]): Lat, Lon

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top, padding_bottom)
    """
    input_resolution = [2] + list(input_resolution)
    window_size = [2] + list(window_size)
    padding = get_pad3d(input_resolution, window_size)
    return padding[: 4]


class SphericalPad2d(nn.Module):
    """Pad latitude with zeros and longitude circularly for global grids."""

    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        if len(padding) != 4:
            raise ValueError(f'Expected 4 padding values, got {padding}.')
        self.padding = tuple(int(p) for p in padding)

    def forward(self, x):
        left, right, top, bottom = self.padding
        if top or bottom:
            x = F.pad(x, (0, 0, top, bottom), mode='constant', value=0)
        if left or right:
            parts = []
            if left:
                parts.append(x[..., -left:])
            parts.append(x)
            if right:
                parts.append(x[..., :right])
            x = torch.cat(parts, dim=-1)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='GELU', drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = getattr(nn, act_layer)()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
        lora_r (int, optional): LoRA rank. Defaults to `8`.
        lora_alpha (int, optional): LoRA alpha. Defaults to `8`.
        lora_dropout (float, optional): LoRA drop-out rate. Defaults to `0.0`.
        lora_steps (int, optional): Maximum number of LoRA roll-out steps. Defaults to `40`.
        lora_mode (str, optional): Mode. `"single"` uses the same LoRA for all roll-out steps,
            and `"all"` uses a different LoRA for every roll-out step. Defaults to `"single"`.
        use_lora (bool, optional): Enable LoRA. By default, LoRA is disabled.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0],
                 lora_r: int = 8,
                 lora_alpha: int = 8,
                 lora_dropout: float = 0.0,
                 lora_steps: int = 40,
                 lora_mode: LoRAMode = "single",
                 use_lora: bool = False,
                 ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        if use_lora:
            self.lora_proj = LoRARollout(
                dim, dim, lora_r, lora_alpha, lora_dropout, lora_steps, lora_mode
            )
            self.lora_qkv = LoRARollout(
                dim, dim * 3, lora_r, lora_alpha, lora_dropout, lora_steps, lora_mode
            )
        else:
            self.lora_proj = lambda *args, **kwargs: 0
            self.lora_qkv = lambda *args, **kwargs: 0

    def forward(self, x, mask=None, rollout_step=0):
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias) + self.lora_qkv(x, rollout_step)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01, device=x.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x) + self.lora_proj(x, rollout_step)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'


class Windowattn_block(nn.Module):
    def __init__(self, dim, window_size, num_heads=1, mlp_ratio=4., 
                qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                act_layer='GELU', norm_layer=nn.LayerNorm,
                attn_type="windowattn", pre_norm=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.pre_norm = pre_norm
        self.attn_type = attn_type
        if "save_attn" in kwargs:
            self.save_attn = kwargs['save_attn']
        else:
            self.save_attn = False

        self.norm = getattr(nn, norm_layer)(dim)
        if attn_type == "windowattn":
            if "shift_size" not in kwargs:
                shift_size = [0, 0, 0]
            else:
                shift_size = kwargs["shift_size"]
            if "dilated_size" in kwargs:
                dilated_size = kwargs["dilated_size"]
            else:
                dilated_size = [1, 1, 1]
            self.attn = SD_attn(
                dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop=attn_drop, proj_drop=drop, shift_size=shift_size, dilated_size=dilated_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = getattr(nn, norm_layer)(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, rollout_step=None):
        shortcut = x

        if self.pre_norm:
            if self.attn_type == "windowattn":
                x, save_attn = self.attn(self.norm(x))
            else:
                x = self.attn(self.norm(x))
            
            x = shortcut + self.drop_path(x)
        else:
            if self.attn_type == "windowattn":
                x, save_attn = self.attn(x)
            else:
                x = self.attn(x)
            
            x = self.norm(shortcut + self.drop_path(x))

        if self.pre_norm:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        
        if self.attn_type == "windowattn" and self.save_attn:
            return x, save_attn
        else:
            return x


class FlashSelfMHAModified(nn.Module):
    """
    self-attention with flashattention
    """
    def __init__(self,
                 dim,
                 num_heads,
                 qkv_bias=True,
                 qk_norm=False,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 device=None,
                 dtype=None,
                 norm_layer='LayerNorm',
                 ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if FlashSelfAttention is None:
            raise ImportError('flash_attn is required to instantiate FlashSelfMHAModified.')
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, (
            "Only support head_dim <= 128 and divisible by 8"
        )

        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias, **factory_kwargs)
        norm = getattr(nn, norm_layer)
        self.q_norm = norm(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = norm(self.head_dim, elementwise_affine=True, eps=1e-6) if qk_norm else nn.Identity()
        self.inner_attn = FlashSelfAttention(attention_dropout=attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        """
        b, s, d = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q).half()
        k = self.k_norm(k).half()

        qkv = torch.stack([q, k, v], dim=2)
        context = self.inner_attn(qkv)
        out = self.out_proj(context.view(b, s, d))
        out = self.proj_drop(out)

        return out


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (str, optional): Activation layer. Default: GELU
        norm_layer (str, optional): Normalization layer.  Default: LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer='GELU', norm_layer='LayerNorm', pretrained_window_size=0,
                 attn_type ='sd_attn',
                 lora_steps: int = 40,
                 lora_mode: LoRAMode = "single",
                 use_lora: bool = False,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = getattr(nn, norm_layer)(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            lora_steps=lora_steps,
            use_lora=use_lora,
            lora_mode=lora_mode,
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = getattr(nn, norm_layer)(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, rollout_step=0):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(
            x_windows,
            mask=self.attn_mask,
            rollout_step=rollout_step,
        )

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (str, optional): Normalization layer.  Default: LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer='LayerNorm'):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = getattr(nn, norm_layer)(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (str, optional): Normalization layer. Default: LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer='LayerNorm', downsample=None, use_checkpoint=False,
                 layer_type='window_block',
                 pretrained_window_size=0,
                 lora_steps: int = 40,
                 lora_mode: LoRAMode = "single",
                 use_lora: bool = False,
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            if layer_type == "window_block":
                block = Windowattn_block(
                        dim=dim,
                        window_size=window_size,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                    )
            elif layer_type == "swin_block":
                block = Windowattn_block(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    shift_size=[0,0] if i%2==0 else [i//2 for i in window_size]
                )

            self.blocks.append(block)                        

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, rollout_step=0):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, rollout_step=rollout_step, use_reentrant=False)
            else:
                x = blk(x, rollout_step=rollout_step)
        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class DownBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, num_groups: int, num_residuals: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        res = x + shortcut
        if h % 2 != 0:
            res = res[:, :, :-1, :]
        if w % 2 != 0:
            res = res[:, :, :, :-1]
        return res


class UpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)

        blk = []
        for i in range(num_residuals):
            blk.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1))
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)

        shortcut = x

        x = self.b(x)

        return x + shortcut


class UTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_groups,
        input_resolution,
        num_heads,
        window_size=7,
        depths=[2, 2, 18, 2],
        num_layers=4,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0,
        drop_path_rate=0,
        attn_drop=0,
        use_checkpoint=True,
        pretrained_window_size=0,
        norm_layer='LayerNorm',
        use_downsample=False,
        lora_steps: int = 40,
        layer_block_type='sd_attn',
        stride=[2, 2],
        lora_mode: LoRAMode = "single",
        use_lora: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.num_levels = len(depths)
        attn_type = kwargs.get('attn_type')
        if attn_type == 'sd_attn':
            layer_block_type = attn_type
        elif attn_type not in (None, 'basic'):
            raise ValueError(f'Unsupported attn_type: {attn_type}')

        padding = get_pad2d(input_resolution, to_2tuple(window_size))
        padding_left, padding_right, padding_top, padding_bottom = padding
        self.padding = padding
        self.pad = SphericalPad2d(padding)
        input_resolution = list(input_resolution)
        input_resolution[0] = input_resolution[0] + padding_top + padding_bottom
        input_resolution[1] = input_resolution[1] + padding_left + padding_right
        self.input_resolution = input_resolution
        self.layer_block_type = layer_block_type
        self.window_size = window_size

        self.down = DownBlock(embed_dim, embed_dim, num_groups)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        layers = []
        for i in range(num_layers):
            if use_downsample:
                downsample = PatchMerging if (i < num_layers - 1) else None
            else:
                downsample = None

            if self.layer_block_type == 'sd_attn':
                window_size = (
                    [input_resolution[0] // stride[-2], input_resolution[1] // stride[-1]]
                    if i == 0
                    else self.window_size
                )

            layer_num_heads = num_heads if isinstance(num_heads, int) else num_heads[i]
            layers.append(
                BasicLayer(
                    dim=int(embed_dim),
                    input_resolution=input_resolution,
                    depth=depths[i],
                    num_heads=layer_num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    downsample=downsample,
                    use_checkpoint=use_checkpoint,
                    pretrained_window_size=pretrained_window_size,
                    use_lora=use_lora,
                    lora_steps=lora_steps,
                    lora_mode=lora_mode,
                    layer_type="window_block" if i == 0 else "swin_block",
                )
            )

        self.layers = nn.Sequential(*layers)

        self.up = UpBlock(embed_dim * 2, embed_dim, num_groups)

    def forward(self, x, rollout_step=0):

        B, C, Lat, Lon = x.shape
        padding_left, padding_right, padding_top, padding_bottom = self.padding
        x = checkpoint.checkpoint(self.down, x, use_reentrant=False)

        shortcut = x

        x = self.pad(x)
        _, _, pad_lat, pad_lon = x.shape

        if self.layer_block_type == 'sd_attn':
            x = rearrange(x, 'b c h w -> b h w c')
        else:
            x = rearrange(x, 'b c h w -> b ( h w ) c')

        for blk in self.layers:
            x = checkpoint.checkpoint(blk, x, rollout_step=rollout_step, use_reentrant=False)

        if self.layer_block_type == 'sd_attn':
            x = rearrange(x, 'b h w c -> b c h w', h=self.input_resolution[0], w=self.input_resolution[1])
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_resolution[0], w=self.input_resolution[1])

        x = x[:, :, padding_top: pad_lat - padding_bottom, padding_left: pad_lon - padding_right]

        x = torch.cat([shortcut, x], dim=1)

        x = checkpoint.checkpoint(self.up, x, use_reentrant=False)
        return x
