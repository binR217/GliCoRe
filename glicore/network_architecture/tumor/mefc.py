"""Multi-scale Evidential Frequency Calibration (MEFC)."""

from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from glicore.network_architecture.dynunet_block import UnetOutBlock
from glicore.network_architecture.tumor.faec import FAEC
from glicore.network_architecture.tumor.fece import FECE


class FrequencyAdaptiveTemperatureBlock(nn.Module):
    """Predict a spatial temperature map from spectrally weighted features."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.frequency_weight = nn.Parameter(
            torch.ones(1, in_channels, 1, 1, 1)
        )
        self.projection = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfftn(
            feature.float(), dim=(2, 3, 4), norm="ortho"
        )
        filtered = torch.fft.irfftn(
            spectrum * self.frequency_weight,
            s=feature.shape[2:],
            dim=(2, 3, 4),
            norm="ortho",
        )
        return self.projection(filtered.to(feature.dtype))


class EvidentialPredictionHead(nn.Module):
    """Segmentation logits plus a frequency-adaptive evidential temperature."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.segmentation_head = UnetOutBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
        )
        self.temperature_head = FrequencyAdaptiveTemperatureBlock(in_channels)

    def forward(
        self, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.segmentation_head(feature)
        temperature = torch.clamp(
            F.softplus(self.temperature_head(feature)) + 1e-2,
            max=5.0,
        )
        evidence = F.softplus(logits / temperature) + 1e-4
        return evidence, logits, temperature


class MEFC(nn.Module):
    """Calibrate full-resolution frequency energy using E2/E3 evidence."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        dec1_channels: int,
        dec2_channels: int,
        patch_size: int = 8,
        full_resolution_size: Union[int, Sequence[int]] = 128,
        target_class_index: int = 3,
        background_index: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(full_resolution_size, int):
            last_axis_size = full_resolution_size
        else:
            if len(full_resolution_size) != 3:
                raise ValueError("full_resolution_size must have three values")
            last_axis_size = int(full_resolution_size[-1])

        self.evidence_head_e2 = EvidentialPredictionHead(
            dec1_channels, num_classes
        )
        self.evidence_head_e3 = EvidentialPredictionHead(
            dec2_channels, num_classes
        )
        frequency_bins = last_axis_size // 2 + 1
        self.faec = FAEC(
            channels=channels,
            patch_size=patch_size,
            frequency_bins=frequency_bins,
        )
        self.fece = FECE(
            channels=channels,
            num_classes=num_classes,
            target_class_index=target_class_index,
            background_index=background_index,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Preserve compatibility with checkpoints created before the paper's
        # FAEC/FECE module boundary was reflected in the public implementation.
        for name in ("W_low_raw", "W_high_raw"):
            old_key = prefix + "fece." + name
            new_key = prefix + "faec." + name
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def construct_evidence(
        self,
        dec1_feature: torch.Tensor,
        dec2_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        evidence_2, _, _ = self.evidence_head_e2(dec1_feature)
        evidence_3, _, _ = self.evidence_head_e3(dec2_feature)
        return evidence_2, evidence_3

    def forward(
        self,
        full_resolution_feature: torch.Tensor,
        dec1_feature: torch.Tensor,
        dec2_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence_2, evidence_3 = self.construct_evidence(
            dec1_feature, dec2_feature
        )
        calibrated, low_response, high_response = self.faec(
            full_resolution_feature
        )
        refined = self.fece(
            calibrated,
            low_response,
            high_response,
            evidence_3=evidence_3,
            evidence_2=evidence_2,
        )
        return refined, evidence_2, evidence_3
