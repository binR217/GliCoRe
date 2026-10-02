"""Frequency-Aware Energy Calibration (FAEC)."""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FAEC(nn.Module):
    """Calibrate local energy and construct complementary spectral responses."""

    def __init__(
        self,
        channels: int = 16,
        patch_size: int = 8,
        frequency_bins: int = 65,
    ):
        super().__init__()
        if frequency_bins < 2:
            raise ValueError("frequency_bins must be at least 2")
        self.patch_size = patch_size
        self.alpha = nn.Parameter(torch.tensor(2.0))
        self.beta = nn.Parameter(torch.tensor(0.1))

        w_low = torch.zeros(1, channels, 1, 1, frequency_bins)
        w_low[..., : frequency_bins // 3] = 2.0
        self.W_low_raw = nn.Parameter(w_low)

        w_high = torch.full((1, channels, 1, 1, frequency_bins), 2.0)
        w_high[..., : frequency_bins // 3] = -2.0
        self.W_high_raw = nn.Parameter(w_high)

    def forward(
        self, f: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, D, H, W = f.shape
        P = self.patch_size

        # Non-overlapping local cubes: (B,C,D,H,W) to
        # (B,C,nD,nH,nW,P,P,P).
        f_patches = f.unfold(2, P, P).unfold(3, P, P).unfold(4, P, P)
        nD, nH, nW = f_patches.shape[2], f_patches.shape[3], f_patches.shape[4]

        # Per-channel low-frequency energy ratio in every cube.
        f_fft = torch.fft.rfftn(f_patches.float(), dim=(5, 6, 7), norm='ortho')
        amp = torch.abs(f_fft)
        freq_len = amp.shape[-1]
        low_cut = max(1, freq_len // 3)
        amp_low = amp[..., :low_cut]
        energy_low = (amp_low ** 2).sum(dim=(5, 6, 7))
        energy_total = (amp ** 2).sum(dim=(5, 6, 7)) + 1e-8
        freq_ratio = energy_low / energy_total  # (B,C,nD,nH,nW)

        # Restore the local ratios to the full feature resolution.
        freq_map = F.interpolate(
            freq_ratio.view(B * C, 1, nD, nH, nW),
            size=(D, H, W), mode='trilinear', align_corners=False
        ).view(B, C, D, H, W)

        # Bounded local energy calibration from Eq. (6) of the paper.
        calibration = torch.tanh(self.alpha * (freq_map - 0.5))
        calibrated = f * (1.0 + self.beta * calibration)

        # Learned spectral responses from the calibrated full-resolution feature.
        spectrum = torch.fft.rfftn(
            calibrated.float(), dim=(2, 3, 4), norm="ortho"
        )
        W_low = torch.sigmoid(self.W_low_raw)
        W_high = torch.sigmoid(self.W_high_raw)
        f_low = torch.fft.irfftn(
            spectrum * W_low,
            s=(D, H, W),
            dim=(2, 3, 4),
            norm="ortho",
        ).to(calibrated.dtype)
        f_high = torch.fft.irfftn(
            spectrum * W_high,
            s=(D, H, W),
            dim=(2, 3, 4),
            norm="ortho",
        ).to(calibrated.dtype)
        return calibrated, f_low, f_high
