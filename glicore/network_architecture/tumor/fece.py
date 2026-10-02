"""Frequency-Evidence Cross-Enhancement (FECE).

FAEC supplies the calibrated feature and its complementary spectral responses.
FECE uses evidential uncertainty to route those responses. Coarse evidence is
the default signal, while fine evidence only corrects it inside target-class
regions and foreground boundaries.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FECE(nn.Module):
    def __init__(
            self,
            channels: int = 16,
            num_classes: int = 4,
            target_class_index: int = 3,
            background_index: int = 0,
            target_correction_init: float = 0.25,
            target_correction_min: float = 0.0,
            target_correction_max: float = 0.45,
            target_mask_threshold: float = 0.35,
            target_mask_scale: float = 8.0,
            foreground_boundary_correction_init: float = 0.12,
            foreground_boundary_correction_min: float = 0.0,
            foreground_boundary_correction_max: float = 0.25,
            foreground_boundary_threshold: float = 0.12,
            foreground_boundary_scale: float = 10.0,
            foreground_boundary_kernel_size: int = 3,
            total_correction_max: float = 0.5,
    ):
        super().__init__()
        if not 0 <= target_class_index < num_classes:
            raise ValueError("target_class_index must be within [0, num_classes)")
        if not 0 <= background_index < num_classes:
            raise ValueError("background_index must be within [0, num_classes)")
        if foreground_boundary_kernel_size < 3 or foreground_boundary_kernel_size % 2 == 0:
            raise ValueError("foreground_boundary_kernel_size must be an odd integer greater than or equal to 3")
        if total_correction_max <= 0:
            raise ValueError("total_correction_max must be positive")

        self.unc_conv = nn.Conv3d(1, channels, kernel_size=1)
        nn.init.constant_(self.unc_conv.weight, 0.1)
        nn.init.zeros_(self.unc_conv.bias)

        self.gamma = nn.Parameter(torch.tensor(0.1))
        self.num_classes = num_classes
        self.target_class_index = target_class_index
        self.background_index = background_index
        self.target_correction_min = target_correction_min
        self.target_correction_max = target_correction_max
        self.target_mask_threshold = target_mask_threshold
        self.target_mask_scale = target_mask_scale
        self.foreground_boundary_correction_min = foreground_boundary_correction_min
        self.foreground_boundary_correction_max = foreground_boundary_correction_max
        self.foreground_boundary_threshold = foreground_boundary_threshold
        self.foreground_boundary_scale = foreground_boundary_scale
        self.foreground_boundary_kernel_size = foreground_boundary_kernel_size
        self.total_correction_max = total_correction_max

        correction_range = target_correction_max - target_correction_min
        if correction_range <= 0:
            raise ValueError("target_correction_max must be greater than target_correction_min")
        init_norm = (target_correction_init - target_correction_min) / correction_range
        init_norm = min(max(init_norm, 1e-4), 1.0 - 1e-4)
        init_logit = torch.log(torch.tensor(init_norm / (1.0 - init_norm), dtype=torch.float32))
        self.target_correction_logit = nn.Parameter(init_logit)

        boundary_correction_range = foreground_boundary_correction_max - foreground_boundary_correction_min
        if boundary_correction_range <= 0:
            raise ValueError("foreground_boundary_correction_max must be greater than foreground_boundary_correction_min")
        boundary_init_norm = (
            foreground_boundary_correction_init
            - foreground_boundary_correction_min
        ) / boundary_correction_range
        boundary_init_norm = min(max(boundary_init_norm, 1e-4), 1.0 - 1e-4)
        boundary_init_logit = torch.log(
            torch.tensor(
                boundary_init_norm / (1.0 - boundary_init_norm),
                dtype=torch.float32,
            )
        )
        self.foreground_boundary_correction_logit = nn.Parameter(
            boundary_init_logit
        )

    @property
    def target_correction_strength(self) -> torch.Tensor:
        correction_range = self.target_correction_max - self.target_correction_min
        return self.target_correction_min + correction_range * torch.sigmoid(self.target_correction_logit)

    @property
    def foreground_boundary_correction_strength(self) -> torch.Tensor:
        correction_range = self.foreground_boundary_correction_max - self.foreground_boundary_correction_min
        return self.foreground_boundary_correction_min + correction_range * torch.sigmoid(
            self.foreground_boundary_correction_logit
        )

    @staticmethod
    def _resize_evidence(evidence: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
        if evidence.shape[2:] == size:
            return evidence
        return F.interpolate(evidence, size=size, mode='trilinear', align_corners=False)

    def _uncertainty_from_evidence(
            self,
            evidence: torch.Tensor,
            size: Tuple[int, int, int],
    ) -> torch.Tensor:
        evidence_up = self._resize_evidence(evidence, size)
        strength = (evidence_up + 1.0).sum(dim=1, keepdim=True)
        return self.num_classes / (strength + 1e-8)

    def _target_candidate_mask(
            self,
            evidence_2: torch.Tensor,
            size: Tuple[int, int, int],
    ) -> torch.Tensor:
        evidence_2_up = self._resize_evidence(evidence_2, size)
        alpha_2 = evidence_2_up + 1.0
        class_prob_2 = alpha_2 / (alpha_2.sum(dim=1, keepdim=True) + 1e-8)
        target_probability = class_prob_2[
            :, self.target_class_index:self.target_class_index + 1
        ]
        return torch.sigmoid(
            self.target_mask_scale
            * (target_probability - self.target_mask_threshold)
        )

    def _foreground_boundary_mask(
            self,
            evidence_2: torch.Tensor,
            size: Tuple[int, int, int],
    ) -> torch.Tensor:
        evidence_2_up = self._resize_evidence(evidence_2, size)
        alpha_2 = evidence_2_up + 1.0
        class_prob_2 = alpha_2 / (alpha_2.sum(dim=1, keepdim=True) + 1e-8)
        background_prob = class_prob_2[:, self.background_index:self.background_index + 1]
        foreground_prob = 1.0 - background_prob

        padding = self.foreground_boundary_kernel_size // 2
        dilated = F.max_pool3d(
            foreground_prob,
            kernel_size=self.foreground_boundary_kernel_size,
            stride=1,
            padding=padding,
        )
        eroded = -F.max_pool3d(
            -foreground_prob,
            kernel_size=self.foreground_boundary_kernel_size,
            stride=1,
            padding=padding,
        )
        boundary = torch.clamp(dilated - eroded, min=0.0, max=1.0)
        return torch.sigmoid(self.foreground_boundary_scale * (boundary - self.foreground_boundary_threshold))

    def _fuse_uncertainty(
            self,
            evidence_3: torch.Tensor,
            evidence_2: Optional[torch.Tensor],
            size: Tuple[int, int, int],
    ) -> torch.Tensor:
        uncertainty_3 = self._uncertainty_from_evidence(evidence_3, size)
        if evidence_2 is None:
            return uncertainty_3

        uncertainty_2 = self._uncertainty_from_evidence(evidence_2, size)
        target_mask = self._target_candidate_mask(evidence_2, size)
        foreground_boundary_mask = self._foreground_boundary_mask(evidence_2, size)

        target_strength = self.target_correction_strength.to(
            device=uncertainty_2.device,
            dtype=uncertainty_2.dtype,
        )
        boundary_strength = self.foreground_boundary_correction_strength.to(
            device=uncertainty_2.device,
            dtype=uncertainty_2.dtype,
        )
        total_mask = (
            target_strength * target_mask
            + boundary_strength * foreground_boundary_mask
        )
        total_mask = torch.clamp(total_mask, min=0.0, max=self.total_correction_max)
        return uncertainty_3 + total_mask * (uncertainty_2 - uncertainty_3)

    def forward(
            self,
            f: torch.Tensor,
            f_low: torch.Tensor,
            f_high: torch.Tensor,
            evidence_3: torch.Tensor,
            evidence_2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, D, H, W = f.shape

        uncertainty = self._fuse_uncertainty(evidence_3, evidence_2, (D, H, W))
        unc_emb = self.unc_conv(uncertainty)

        gate_low = torch.sigmoid(unc_emb)
        gate_high = 1.0 - gate_low

        f_mixed = gate_low * f_low + gate_high * f_high
        return f + self.gamma * f_mixed
