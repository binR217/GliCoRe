"""Position-Aware Case-context Enhanced Relational Learning (PACER)."""

import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.act(self.norm(self.depthwise(x))))


class WholeCaseEncoder(nn.Module):
    def __init__(
        self,
        image_channels: int,
        memory_channels: int,
        num_foreground_classes: int,
    ):
        super().__init__()
        input_channels = image_channels + 1
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(8), 8),
            nn.GELU(),
        )
        self.stages = nn.Sequential(
            DepthwiseSeparableBlock(8, 16, stride=2),
            DepthwiseSeparableBlock(16, memory_channels, stride=2),
            DepthwiseSeparableBlock(memory_channels, memory_channels, stride=2),
        )
        self.occupancy_head = nn.Conv3d(
            memory_channels, num_foreground_classes, kernel_size=1
        )

    def forward(
        self, x: torch.Tensor, compute_occupancy: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        memory = self.stages(self.stem(x))
        occupancy = self.occupancy_head(memory) if compute_occupancy else None
        return memory, occupancy


class PositionBiasedCrossAttention(nn.Module):
    def __init__(
        self,
        local_channels: int = 32,
        memory_channels: int = 24,
        num_heads: int = 2,
        memory_size: int = 8,
        nominal_patch_size: Sequence[int] = (128, 128, 128),
    ):
        super().__init__()
        if memory_channels % num_heads != 0:
            raise ValueError("memory_channels must be divisible by num_heads")
        self.local_channels = local_channels
        self.memory_channels = memory_channels
        self.num_heads = num_heads
        self.head_dim = memory_channels // num_heads
        self.memory_size = int(memory_size)
        self.register_buffer(
            "nominal_patch_size",
            torch.as_tensor(nominal_patch_size, dtype=torch.float32),
            persistent=False,
        )

        self.query = nn.Linear(local_channels, memory_channels, bias=False)
        self.key = nn.Linear(memory_channels, memory_channels, bias=False)
        self.value = nn.Linear(memory_channels, memory_channels, bias=False)
        self.output = nn.Conv3d(
            memory_channels, local_channels, kernel_size=1, bias=False
        )
        nn.init.kaiming_normal_(self.output.weight, mode="fan_out", nonlinearity="linear")

        coords = torch.linspace(-1.0, 1.0, self.memory_size)
        grid = torch.stack(
            torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1
        ).reshape(-1, 3)
        self.register_buffer("token_coordinates", grid, persistent=False)

    def _patch_position_bias(
        self,
        patch_center: torch.Tensor,
        case_shape: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = case_shape.to(device=patch_center.device, dtype=dtype).clamp_min(1.0)
        patch = self.nominal_patch_size.to(device=patch_center.device, dtype=dtype)
        half_span = (patch / shape).clamp(max=1.0)
        denominator = torch.maximum(
            half_span,
            half_span.new_full(half_span.shape, 1.0 / self.memory_size),
        )
        key_coords = self.token_coordinates.to(
            device=patch_center.device, dtype=dtype
        )
        relative = (
            key_coords[None, :, :] - patch_center.to(dtype=dtype)[:, None, :]
        ) / denominator[:, None, :]
        return -0.5 * torch.sum(relative.square(), dim=-1)

    def forward(
        self,
        local_feature: torch.Tensor,
        memory: torch.Tensor,
        patch_center: torch.Tensor,
        case_shape: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pooled = F.adaptive_avg_pool3d(
            local_feature, (self.memory_size,) * 3
        )
        local_tokens = pooled.flatten(2).transpose(1, 2)
        memory_tokens = memory.flatten(2).transpose(1, 2)

        batch, query_count, _ = local_tokens.shape
        key_count = memory_tokens.shape[1]
        q = self.query(local_tokens).view(
            batch, query_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.key(memory_tokens).view(
            batch, key_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(memory_tokens).view(
            batch, key_count, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.memory_channels
        )
        position_bias = self._patch_position_bias(
            patch_center, case_shape, scores.dtype
        )
        scores = scores + position_bias[:, None, None, :]

        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v).transpose(1, 2).reshape(
            batch, query_count, self.memory_channels
        )
        context = context.transpose(1, 2).reshape(
            batch,
            self.memory_channels,
            self.memory_size,
            self.memory_size,
            self.memory_size,
        )
        context = F.interpolate(
            context,
            size=local_feature.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        residual = self.output(context)
        return residual, attention


class PACER(nn.Module):
    def __init__(
        self,
        image_channels: int,
        local_channels: int,
        num_classes: int,
        memory_channels: int = 24,
        memory_size: int = 8,
        nominal_patch_size: Sequence[int] = (128, 128, 128),
    ):
        super().__init__()
        self.global_encoder = WholeCaseEncoder(
            image_channels=image_channels,
            memory_channels=memory_channels,
            num_foreground_classes=num_classes - 1,
        )
        self.aligner = PositionBiasedCrossAttention(
            local_channels=local_channels,
            memory_channels=memory_channels,
            num_heads=2,
            memory_size=memory_size,
            nominal_patch_size=nominal_patch_size,
        )

    def encode_global(
        self, global_input: torch.Tensor, compute_occupancy: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.global_encoder(global_input, compute_occupancy=compute_occupancy)

    def align_local(
        self,
        dec1: torch.Tensor,
        memory: torch.Tensor,
        patch_center: torch.Tensor,
        case_shape: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        residual, attention = self.aligner(
            dec1, memory, patch_center, case_shape
        )
        return dec1 + residual, {
            "context_residual": residual,
            "attention": attention,
        }


class BoundaryRelationHead(nn.Module):
    def __init__(self, in_channels: int = 16, embedding_channels: int = 8):
        super().__init__()
        self.projection = nn.Conv3d(
            in_channels, embedding_channels, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(x), dim=1, eps=1e-6)
