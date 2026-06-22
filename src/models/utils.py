import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim = -1)
        return torch.matmul(attn, x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.,
        patch_merge_layer=None,
        patch_merge_num_tokens=8,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])

        self.patch_merge_layer_index = default(patch_merge_layer, depth // 2) - 1
        self.patch_merger = PatchMerger(dim=dim, num_tokens_out=patch_merge_num_tokens)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for index, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x

            if index == self.patch_merge_layer_index:
                x = self.patch_merger(x)

        return self.norm(x)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(8, channels)


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        self.cached_penc = self.cached_penc.reshape(*(self.cached_penc.shape[:-1]), -1, 2)
        sin_data = torch.cat(
            (
                -self.cached_penc[:, :, :, :, :, 0].unsqueeze(-1),
                self.cached_penc[:, :, :, :, :, 0].unsqueeze(-1),
            ),
            dim=-1,
        )
        cos_data = self.cached_penc[:, :, :, :, :, 1].unsqueeze(-1)

        origin_shape = tensor.shape

        tensor = tensor.reshape(*(tensor.shape[:-1]), -1, 2)
        tensor_flip = torch.flip(tensor, dims=[-1])
        res = tensor * cos_data + tensor_flip * sin_data
        return res.reshape(*origin_shape)


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class Summer(nn.Module):
    def __init__(self, penc):
        """
        :param model: The type of positional encoding to run the summer on.
        """
        super(Summer, self).__init__()
        self.penc = penc

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        penc = self.penc(tensor)
        assert (
            tensor.size() == penc.size()
        ), "The original tensor size {} and the positional encoding tensor size {} must match!".format(
            tensor.size, penc.size
        )
        tensor = tensor.reshape(*(tensor.shape[:-1]), 2)
        tensor_flip = torch.flip(tensor, dims=[-1])
        tensor_flip = tensor_flip * torch.Tensor([-1, 1], device=tensor.device)
        
        return tensor + penc


class rope2(nn.Module):
    def __init__(self, shape, dim, origin_shape=[0,0]) -> None:
        super().__init__()
        
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        
        if origin_shape[0] > 0:
            coords_0 = coords_0 / (shape[0] - 1) * (origin_shape[0] - 1)
            coords_1 = coords_1 / (shape[1] - 1) * (origin_shape[1] - 1)
        coords = torch.stack(torch.meshgrid([coords_0, coords_1], indexing="ij")).reshape(2, -1)

        half_size = dim // 2
        self.dim1_size = half_size // 2
        self.dim2_size = half_size - half_size // 2
        freq_seq1 = torch.arange(0, self.dim1_size) / self.dim1_size
        freq_seq2 = torch.arange(0, self.dim2_size) / self.dim2_size
        inv_freq1 = 10000 ** -freq_seq1
        inv_freq2 = 10000 ** -freq_seq2

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1    
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq2     

        self.sin1 = torch.sin(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.cos1 = torch.cos(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.sin2 = torch.sin(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.cos2 = torch.cos(sinusoid2).reshape(*shape, sinusoid2.shape[-1])


    def forward(self, x):

        self.sin1 = self.sin1.to(x)
        self.cos1 = self.cos1.to(x)
        self.sin2 = self.sin2.to(x)
        self.cos2 = self.cos2.to(x)

        x11, x21, x12, x22 = x.split([self.dim1_size, self.dim2_size, \
                                        self.dim1_size, self.dim2_size], dim=-1)
        
        res = torch.cat([x11 * self.cos1 - x12 * self.sin1, x21 * self.cos2 - x22 * self.sin2, \
                        x12 * self.cos1 + x11 * self.sin1, x22 * self.cos2 + x21 * self.sin2], dim=-1)

        return res


class rope3(nn.Module):
    def __init__(self, shape, dim) -> None:
        super().__init__()
        
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        coords_2 = torch.arange(shape[2])
        coords = torch.stack(torch.meshgrid([coords_0, coords_1, coords_2], indexing="ij")).reshape(3, -1)

        half_size = dim // 2
        self.dim1_2_size = half_size // 3
        self.dim3_size = half_size - half_size // 3 * 2
        freq_seq1_2 = torch.arange(0, self.dim1_2_size) / self.dim1_2_size
        freq_seq3 = torch.arange(0, self.dim3_size) / self.dim3_size
        inv_freq1_2 = 10000 ** -freq_seq1_2
        inv_freq3 = 10000 ** -freq_seq3

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1_2    
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq1_2    
        sinusoid3 = coords[2].unsqueeze(-1) * inv_freq3    

        self.sin1 = torch.sin(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.cos1 = torch.cos(sinusoid1).reshape(*shape, sinusoid1.shape[-1])
        self.sin2 = torch.sin(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.cos2 = torch.cos(sinusoid2).reshape(*shape, sinusoid2.shape[-1])
        self.sin3 = torch.sin(sinusoid3).reshape(*shape, sinusoid3.shape[-1])
        self.cos3 = torch.cos(sinusoid3).reshape(*shape, sinusoid3.shape[-1])


    def forward(self, x):

        self.sin1 = self.sin1.to(x)
        self.cos1 = self.cos1.to(x)
        self.sin2 = self.sin2.to(x)
        self.cos2 = self.cos2.to(x)
        self.sin3 = self.sin3.to(x)
        self.cos3 = self.cos3.to(x)

        split_sizes = [
            self.dim1_2_size,
            self.dim1_2_size,
            self.dim3_size,
            self.dim1_2_size,
            self.dim1_2_size,
            self.dim3_size,
        ]
        x11, x21, x31, x12, x22, x32 = x.split(split_sizes, dim=-1)

        res = torch.cat(
            [
                x11 * self.cos1 - x12 * self.sin1,
                x21 * self.cos2 - x22 * self.sin2,
                x31 * self.cos3 - x32 * self.sin3,
                x12 * self.cos1 + x11 * self.sin1,
                x22 * self.cos2 + x21 * self.sin2,
                x32 * self.cos3 + x31 * self.sin3,
            ],
            dim=-1,
        )

        return res


class rope3_maskflatten(nn.Module):
    def __init__(self, shape, dim) -> None:
        super().__init__()
        
        coords_0 = torch.arange(shape[0])
        coords_1 = torch.arange(shape[1])
        coords_2 = torch.arange(shape[2])
        coords = torch.stack(torch.meshgrid([coords_0, coords_1, coords_2], indexing="ij")).reshape(3, -1)

        half_size = dim // 2
        self.dim1_2_size = half_size // 3
        self.dim3_size = half_size - half_size // 3 * 2
        freq_seq1_2 = torch.arange(0, self.dim1_2_size) / self.dim1_2_size
        freq_seq3 = torch.arange(0, self.dim3_size) / self.dim3_size
        inv_freq1_2 = 10000 ** -freq_seq1_2
        inv_freq3 = 10000 ** -freq_seq3

        sinusoid1 = coords[0].unsqueeze(-1) * inv_freq1_2    
        sinusoid2 = coords[1].unsqueeze(-1) * inv_freq1_2    
        sinusoid3 = coords[2].unsqueeze(-1) * inv_freq3    

        self.sin1 = torch.sin(sinusoid1).reshape(-1, sinusoid1.shape[-1])
        self.cos1 = torch.cos(sinusoid1).reshape(-1, sinusoid1.shape[-1])
        self.sin2 = torch.sin(sinusoid2).reshape(-1, sinusoid2.shape[-1])
        self.cos2 = torch.cos(sinusoid2).reshape(-1, sinusoid2.shape[-1])
        self.sin3 = torch.sin(sinusoid3).reshape(-1, sinusoid3.shape[-1])
        self.cos3 = torch.cos(sinusoid3).reshape(-1, sinusoid3.shape[-1])


    def forward(self, x, mask):

        if mask is not None:
            x_mask_indices = mask[0].nonzero(as_tuple=True)
            sin1 = self.sin1.to(x)[x_mask_indices]
            cos1 = self.cos1.to(x)[x_mask_indices]
            sin2 = self.sin2.to(x)[x_mask_indices]
            cos2 = self.cos2.to(x)[x_mask_indices]
            sin3 = self.sin3.to(x)[x_mask_indices]
            cos3 = self.cos3.to(x)[x_mask_indices]
        else:
            sin1 = self.sin1.to(x)
            cos1 = self.cos1.to(x)
            sin2 = self.sin2.to(x)
            cos2 = self.cos2.to(x)
            sin3 = self.sin3.to(x)
            cos3 = self.cos3.to(x)

        split_sizes = [
            self.dim1_2_size,
            self.dim1_2_size,
            self.dim3_size,
            self.dim1_2_size,
            self.dim1_2_size,
            self.dim3_size,
        ]
        x11, x21, x31, x12, x22, x32 = x.split(split_sizes, dim=-1)

        res = torch.cat(
            [
                x11 * cos1 - x12 * sin1,
                x21 * cos2 - x22 * sin2,
                x31 * cos3 - x32 * sin3,
                x12 * cos1 + x11 * sin1,
                x22 * cos2 + x21 * sin2,
                x32 * cos3 + x31 * sin3,
            ],
            dim=-1,
        )

        return res


class RelativePositionalBias(nn.Module):
    def __init__(self, window_size, num_heads=1) -> None:
        super().__init__()

        self.total_window_size = 1
        table_len = 1
        for i in window_size:
            table_len *= 2 * i - 1
            self.total_window_size *= i

        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_len, num_heads))

        coords = []
        for i in window_size:
            coords.append(torch.arange(i))

        coords = torch.stack(torch.meshgrid(coords, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        for i in range(len(window_size)):
            relative_coords[:, :, i] += window_size[i] - 1
        for i in range(len(window_size) - 1):
            table_len = table_len // (2 * window_size[i] - 1)
            relative_coords[:, :, i] *= table_len

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.total_window_size, self.total_window_size, -1)
            
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        x = x + relative_position_bias

        return x



def window_partition(x, window_size: tuple):
    """
    Split a feature map into non-overlapping windows.

    Args:
        x: (B, H, W, C)
        window_size (tuple): window size(Wt, Wh, Ww)

    Returns:
        windows: (num_windows*B, window_size, C)
    """
    if len(window_size) == 3:
        B, T, H, W, C = x.shape
        x = x.view(
            B,
            T // window_size[0],
            window_size[0],
            H // window_size[1],
            window_size[1],
            W // window_size[2],
            window_size[2],
            C,
        )
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
        windows = windows.view(-1, window_size[0], window_size[1], window_size[2], C)
    elif len(window_size) == 2:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size[0], window_size[1], C)
    return windows



def window_reverse(windows, window_size, T=1, H=1, W=1):
    """
    Reconstruct a feature map from windows.

    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    if len(window_size) == 3:
        B = int(
            windows.shape[0] / (T * H * W / window_size[0] / window_size[1] / window_size[2])
        )
        x = windows.view(
            B,
            T // window_size[0],
            H // window_size[1],
            W // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, T, H, W, -1)
    elif len(window_size) == 2:
        B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
        x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
