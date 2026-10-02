"""Standard UNETR++ transformer block for the tumor backbone.

Frequency calibration is implemented by MEFC after the full-resolution
convolutional decoder rather than inside the encoder-decoder backbone.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from glicore.network_architecture.dynunet_block import UnetResBlock


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class TransformerBlock(nn.Module):
    """UNETR++ transformer block with the original paired attention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed: bool = False,
    ) -> None:
        super().__init__()
        if not 0 <= dropout_rate <= 1:
            raise ValueError("dropout_rate should be between 0 and 1")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size))
        self.epa_block = EPA(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_heads=num_heads,
            channel_attn_drop=dropout_rate,
            spatial_attn_drop=dropout_rate,
        )
        self.conv51 = UnetResBlock(
            3,
            hidden_size,
            hidden_size,
            kernel_size=3,
            stride=1,
            norm_name="batch",
        )
        self.conv8 = nn.Sequential(
            nn.Dropout3d(0.1, False),
            nn.Conv3d(hidden_size, hidden_size, kernel_size=1),
        )
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, input_size, hidden_size))
            if pos_embed
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width, depth = x.shape
        tokens = x.reshape(batch, channels, height * width * depth).permute(0, 2, 1)
        if self.pos_embed is not None:
            tokens = tokens + self.pos_embed

        attention = tokens + self.gamma * self.epa_block(self.norm(tokens))
        attention_skip = attention.reshape(
            batch, height, width, depth, channels
        ).permute(0, 4, 1, 2, 3)
        refined = self.conv51(attention_skip)
        return attention_skip + self.conv8(refined)


class EPA(nn.Module):
    """Efficient Paired Attention from the original UNETR++ backbone."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        channel_attn_drop: float = 0.1,
        spatial_attn_drop: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)
        self.EF = nn.Parameter(init_(torch.zeros(input_size, proj_size)))
        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkvv = self.qkvv(x).reshape(
            batch,
            tokens,
            4,
            self.num_heads,
            channels // self.num_heads,
        )
        q_shared, k_shared, v_channel, v_spatial = qkvv.permute(2, 0, 3, 1, 4)

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_channel = v_channel.transpose(-2, -1)
        v_spatial = v_spatial.transpose(-2, -1)
        proj = lambda args: torch.einsum("bhdn,nk->bhdk", *args)
        k_projected, v_spatial_projected = map(
            proj, zip((k_shared, v_spatial), (self.EF, self.EF))
        )

        q_shared = F.normalize(q_shared, dim=-1)
        k_shared = F.normalize(k_shared, dim=-1)

        channel_attention = (
            q_shared @ k_shared.transpose(-2, -1)
        ) * self.temperature
        channel_attention = self.attn_drop(channel_attention.softmax(dim=-1))
        x_channel = (channel_attention @ v_channel).permute(0, 3, 1, 2).reshape(
            batch, tokens, channels
        )

        spatial_attention = (
            q_shared.permute(0, 1, 3, 2) @ k_projected
        ) * self.temperature2
        spatial_attention = self.attn_drop_2(spatial_attention.softmax(dim=-1))
        x_spatial = (
            spatial_attention @ v_spatial_projected.transpose(-2, -1)
        ).permute(0, 3, 1, 2).reshape(batch, tokens, channels)

        return x_channel + x_spatial

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature", "temperature2"}
