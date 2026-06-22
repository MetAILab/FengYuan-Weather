import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from .utransformer import UTransformer


class CubeEmbedding(nn.Module):
    """
    Args:
        img_size: T, Lat, Lon
        patch_size: T, Lat, Lon
    """
    def __init__(self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        assert T == self.img_size[0] and Lat == self.img_size[1] and Lon == self.img_size[2], \
            f"Input image size ({T}*{Lat}*{Lon}) doesn't match model " \
            f"({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x


class FengYuan(nn.Module):
    def __init__(
        self,
        img_size=(2, 721, 1440),
        patch_size=(2, 4, 4),
        in_channels=70,
        out_channels=70,
        embed_dim=1536,
        num_groups=32,
        num_heads=8,
        window_size=7,
        depth=48,
        drop_rate=0,
        drop_path_rate=0.2,
        **kwargs,
    ):
        super().__init__()
        input_resolution = int(img_size[1] / patch_size[1] / 2), int(img_size[2] / patch_size[2] / 2)
        self.cube_embedding = CubeEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.u_transformer = UTransformer(
            embed_dim=embed_dim,
            num_groups=num_groups,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

        self.pred_channels = out_channels
        self.fc = nn.Linear(embed_dim, out_channels * patch_size[1] * patch_size[2])

        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.out_chans = out_channels
        self.img_size = img_size

    def forward(self, x: torch.Tensor, t=None):
        B, _, _, _, _ = x.shape
        _, patch_lat, patch_lon = self.patch_size
        Lat, Lon = self.input_resolution
        Lat, Lon = Lat * 2, Lon * 2

        x = checkpoint(
            self.cube_embedding,
            rearrange(x, 'b t c h w -> b c t h w'),
            use_reentrant=False,
        ).squeeze(2)

        x = self.u_transformer(x)

        x = checkpoint(self.fc, x.permute(0, 2, 3, 1), use_reentrant=False)

        x = x.reshape(B, Lat, Lon, patch_lat, patch_lon, self.out_chans)
        x = x.permute(0, 1, 3, 2, 4, 5)

        x = x.reshape(B, Lat * patch_lat, Lon * patch_lon, self.out_chans)
        x = x.permute(0, 3, 1, 2)

        x = F.interpolate(x, size=self.img_size[1:], mode="bilinear")

        return x
