import math
from functools import reduce
from operator import mul
from typing import Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from glicore.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from glicore.network_architecture.neural_network import SegmentationNetwork
from glicore.network_architecture.tumor.pacer import BoundaryRelationHead, PACER


def _product(values: Sequence[int]) -> int:
    return int(reduce(mul, (int(value) for value in values), 1))


class FATB(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.freq_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1, 1))
        self.conv = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfftn(feature.float(), dim=(2, 3, 4), norm="ortho")
        filtered = torch.fft.irfftn(
            spectrum * self.freq_weight,
            s=feature.shape[2:],
            dim=(2, 3, 4),
            norm="ortho",
        )
        return self.conv(filtered.to(feature.dtype))


class FrequencyAwareEvidenceCalibration(nn.Module):
    def __init__(self, channels: int = 16, patch_size: Union[int, Sequence[int]] = 8):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size,) * 3
        if len(patch_size) != 3 or any(int(value) <= 0 for value in patch_size):
            raise ValueError("patch_size must contain three positive values")
        self.patch_size = tuple(int(value) for value in patch_size)
        self.alpha = nn.Parameter(torch.tensor(2.0))
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = feature.shape
        patch_d, patch_h, patch_w = self.patch_size
        if depth % patch_d or height % patch_h or width % patch_w:
            raise ValueError(
                "FAEC patch size %r must divide feature size %r"
                % (self.patch_size, tuple(feature.shape[2:]))
            )
        patches = (
            feature.unfold(2, patch_d, patch_d)
            .unfold(3, patch_h, patch_h)
            .unfold(4, patch_w, patch_w)
        )
        grid_size = patches.shape[2:5]
        spectrum = torch.fft.rfftn(patches.float(), dim=(5, 6, 7), norm="ortho")
        amplitude = spectrum.abs()
        low_cut = max(1, amplitude.shape[-1] // 3)
        low_energy = amplitude[..., :low_cut].square().sum(dim=(5, 6, 7))
        total_energy = amplitude.square().sum(dim=(5, 6, 7)).clamp_min(1e-8)
        ratio = low_energy / total_energy
        frequency_map = F.interpolate(
            ratio.reshape(batch * channels, 1, *grid_size),
            size=(depth, height, width),
            mode="trilinear",
            align_corners=False,
        ).reshape(batch, channels, depth, height, width)
        calibration = torch.tanh(self.alpha * (frequency_map - 0.5))
        return feature * (1.0 + self.beta * calibration)


class GeneralizedFECE(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        frequency_bins: int,
        background_index: int = 0,
        semantic_correction_init: float = 0.25,
        semantic_correction_max: float = 0.45,
        boundary_correction_init: float = 0.12,
        boundary_correction_max: float = 0.25,
        total_correction_max: float = 0.5,
        boundary_threshold: float = 0.12,
        boundary_scale: float = 10.0,
        reliability_scale: float = 8.0,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and foreground")
        if not 0 <= background_index < num_classes:
            raise ValueError("background_index is outside the class range")
        if frequency_bins < 2:
            raise ValueError("frequency_bins must be at least two")

        low = torch.zeros(1, channels, 1, 1, frequency_bins)
        low[..., : frequency_bins // 3] = 2.0
        high = torch.full((1, channels, 1, 1, frequency_bins), 2.0)
        high[..., : frequency_bins // 3] = -2.0
        self.W_low_raw = nn.Parameter(low)
        self.W_high_raw = nn.Parameter(high)
        self.unc_conv = nn.Conv3d(1, channels, kernel_size=1)
        nn.init.constant_(self.unc_conv.weight, 0.1)
        nn.init.zeros_(self.unc_conv.bias)
        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.num_classes = int(num_classes)
        self.background_index = int(background_index)
        self.semantic_correction_max = float(semantic_correction_max)
        self.boundary_correction_max = float(boundary_correction_max)
        self.total_correction_max = float(total_correction_max)
        self.boundary_threshold = float(boundary_threshold)
        self.boundary_scale = float(boundary_scale)
        self.reliability_scale = float(reliability_scale)
        self.semantic_correction_logit = nn.Parameter(
            self._bounded_logit(semantic_correction_init, semantic_correction_max)
        )
        self.boundary_correction_logit = nn.Parameter(
            self._bounded_logit(boundary_correction_init, boundary_correction_max)
        )

    @staticmethod
    def _bounded_logit(initial: float, maximum: float) -> torch.Tensor:
        if not 0.0 <= initial <= maximum or maximum <= 0.0:
            raise ValueError("correction initialization must lie within its range")
        ratio = min(max(initial / maximum, 1e-4), 1.0 - 1e-4)
        return torch.log(torch.tensor(ratio / (1.0 - ratio), dtype=torch.float32))

    @property
    def semantic_correction_strength(self) -> torch.Tensor:
        return self.semantic_correction_max * torch.sigmoid(
            self.semantic_correction_logit
        )

    @property
    def boundary_correction_strength(self) -> torch.Tensor:
        return self.boundary_correction_max * torch.sigmoid(
            self.boundary_correction_logit
        )

    @staticmethod
    def _resize(tensor: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
        if tensor.shape[2:] == size:
            return tensor
        return F.interpolate(tensor, size=size, mode="trilinear", align_corners=False)

    def _opinion(
        self, evidence: torch.Tensor, size: Tuple[int, int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        evidence = self._resize(evidence, size)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return alpha / strength, self.num_classes / strength

    def _interface_map(self, probability: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool3d(probability, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool3d(-probability, kernel_size=3, stride=1, padding=1)
        interface = (dilated - eroded).clamp(0.0, 1.0).amax(dim=1, keepdim=True)
        raw = torch.sigmoid(
            self.boundary_scale * (interface - self.boundary_threshold)
        )
        baseline = 1.0 / (
            1.0 + math.exp(self.boundary_scale * self.boundary_threshold)
        )
        return ((raw - baseline) / max(1.0 - baseline, 1e-6)).clamp(0.0, 1.0)

    def _fused_uncertainty(
        self,
        evidence_3: torch.Tensor,
        evidence_2: torch.Tensor,
        size: Tuple[int, int, int],
    ) -> torch.Tensor:
        _, uncertainty_3 = self._opinion(evidence_3, size)
        if evidence_2 is None:
            return uncertainty_3
        probability_2, uncertainty_2 = self._opinion(evidence_2, size)

        entropy = -(probability_2.clamp_min(1e-8).log() * probability_2).sum(
            dim=1, keepdim=True
        )
        confidence = (1.0 - entropy / math.log(self.num_classes)).clamp(0.0, 1.0)
        foreground = 1.0 - probability_2[
            :, self.background_index : self.background_index + 1
        ]
        advantage = (
            2.0
            * torch.sigmoid(
                self.reliability_scale * (uncertainty_3 - uncertainty_2)
            )
            - 1.0
        ).clamp(0.0, 1.0)
        semantic_mask = foreground * confidence * advantage
        interface_mask = self._interface_map(probability_2) * (0.25 + 0.75 * advantage)
        correction = (
            self.semantic_correction_strength.to(uncertainty_2) * semantic_mask
            + self.boundary_correction_strength.to(uncertainty_2) * interface_mask
        ).clamp(0.0, self.total_correction_max)
        return uncertainty_3 + correction * (uncertainty_2 - uncertainty_3)

    @staticmethod
    def _resize_window(window: torch.Tensor, frequency_bins: int) -> torch.Tensor:
        if window.shape[-1] == frequency_bins:
            return window
        channels = window.shape[1]
        resized = F.interpolate(
            window.reshape(channels, 1, window.shape[-1]),
            size=frequency_bins,
            mode="linear",
            align_corners=False,
        )
        return resized.reshape(1, channels, 1, 1, frequency_bins)

    def forward(
        self,
        feature: torch.Tensor,
        evidence_3: torch.Tensor,
        evidence_2: torch.Tensor = None,
    ) -> torch.Tensor:
        spatial_size = tuple(int(value) for value in feature.shape[2:])
        spectrum = torch.fft.rfftn(feature.float(), dim=(2, 3, 4), norm="ortho")
        amplitude = spectrum.abs()
        phase = torch.angle(spectrum)
        low_window = torch.sigmoid(
            self._resize_window(self.W_low_raw, amplitude.shape[-1])
        )
        high_window = torch.sigmoid(
            self._resize_window(self.W_high_raw, amplitude.shape[-1])
        )
        phase_factor = torch.exp(1j * phase)
        low = torch.fft.irfftn(
            amplitude * low_window * phase_factor,
            s=spatial_size,
            dim=(2, 3, 4),
            norm="ortho",
        ).to(feature.dtype)
        high = torch.fft.irfftn(
            amplitude * high_window * phase_factor,
            s=spatial_size,
            dim=(2, 3, 4),
            norm="ortho",
        ).to(feature.dtype)
        uncertainty = self._fused_uncertainty(
            evidence_3, evidence_2, spatial_size
        )
        low_gate = torch.sigmoid(self.unc_conv(uncertainty))
        mixed = low_gate * low + (1.0 - low_gate) * high
        return feature + self.gamma * mixed


class UBER(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = int(num_classes)
        self.probe_conv = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        self.spatial_edge = nn.Sequential(
            nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
            ),
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.freq_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1, 1))
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.tensor([5.0]))
        self.mu = nn.Parameter(torch.tensor([0.2]))
        self.lambda_inject = nn.Parameter(torch.tensor([0.5]))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        alpha = F.softplus(self.probe_conv(feature)) + 1.0
        uncertainty = self.num_classes / alpha.sum(dim=1, keepdim=True)
        uncertain_mask = torch.sigmoid(self.gamma * (uncertainty - self.mu))
        spatial = self.spatial_edge(feature)
        spectrum = torch.fft.rfftn(feature.float(), dim=(2, 3, 4), norm="ortho")
        frequency = torch.fft.irfftn(
            spectrum * self.freq_weight,
            s=feature.shape[2:],
            dim=(2, 3, 4),
            norm="ortho",
        ).to(feature.dtype)
        edge = self.fusion_conv(torch.cat((spatial, frequency), dim=1)) * feature
        return feature + self.lambda_inject * uncertain_mask * edge


class AdaTERModule(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.semantic_proj = nn.Sequential(
            nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1),
            nn.InstanceNorm3d(num_classes),
        )
        self.morph_dilation = nn.MaxPool3d(kernel_size=5, stride=1, padding=2)
        self.topology_matrix = nn.Parameter(
            torch.eye(num_classes) + 0.05 * torch.randn(num_classes, num_classes)
        )
        self.feat_reconstruct = nn.Sequential(
            nn.Conv3d(num_classes + in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
            nn.GELU(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        semantic = self.semantic_proj(feature)
        evidence = F.softplus(semantic)
        belief = evidence / (evidence + 1.0)
        envelopes = self.morph_dilation(belief)
        batch, classes, depth, height, width = envelopes.shape
        topology = torch.sigmoid(self.topology_matrix)
        gating = torch.einsum(
            "ij,bjn->bin", topology, envelopes.reshape(batch, classes, -1)
        ).reshape(batch, classes, depth, height, width)
        return self.feat_reconstruct(torch.cat((feature, semantic * gating), dim=1))


class GliCoReGeneralized(SegmentationNetwork):
    def __init__(
        self,
        encoder_class,
        up_block_class,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int],
        feat_size: Sequence[int],
        decoder_sizes: Sequence[Sequence[int]],
        final_upsample: Sequence[int],
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        norm_name: Union[Tuple, str] = "instance",
        depths=None,
        dims=None,
        conv_op=nn.Conv3d,
        do_ds: bool = True,
        enable_pacer: bool = False,
        faec_patch_size: Union[int, Sequence[int]] = 8,
    ):
        super().__init__()
        depths = [3, 3, 3, 3] if depths is None else depths
        dims = [32, 64, 128, 256] if dims is None else dims
        self.do_ds = bool(do_ds)
        self.conv_op = conv_op
        self.num_classes = int(out_channels)
        self.hidden_size = int(hidden_size)
        self.feat_size = tuple(int(value) for value in feat_size)
        self.img_size = tuple(int(value) for value in img_size)
        self.enable_pacer = bool(enable_pacer)
        self.mask_policy = "foreground_interface"

        self.glicore_encoder = encoder_class(
            dims=dims, depths=depths, num_heads=num_heads
        )
        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = up_block_class(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=_product(decoder_sizes[0]),
        )
        self.decoder4 = up_block_class(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=_product(decoder_sizes[1]),
        )
        self.decoder3 = up_block_class(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=_product(decoder_sizes[2]),
        )
        self.decoder2 = up_block_class(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=tuple(int(value) for value in final_upsample),
            norm_name=norm_name,
            out_size=_product(self.img_size),
            conv_decoder=True,
        )

        self.faec_module = FrequencyAwareEvidenceCalibration(
            channels=feature_size, patch_size=faec_patch_size
        )
        self.fece_module = GeneralizedFECE(
            channels=feature_size,
            num_classes=out_channels,
            frequency_bins=self.img_size[-1] // 2 + 1,
        )
        self.uber_module = UBER(feature_size, out_channels)
        self.adater_module = AdaTERModule(feature_size, out_channels)
        self.out1 = UnetOutBlock(3, feature_size, out_channels)
        self.out2 = UnetOutBlock(3, feature_size * 2, out_channels)
        self.out3 = UnetOutBlock(3, feature_size * 4, out_channels)
        self.temp_out1 = FATB(feature_size)
        self.temp_out2 = FATB(feature_size * 2)
        self.temp_out3 = FATB(feature_size * 4)

        if self.enable_pacer:
            self.pacer_module = PACER(
                image_channels=in_channels,
                local_channels=feature_size * 2,
                num_classes=out_channels,
                nominal_patch_size=self.img_size,
            )
            self.pacer_relation_head = BoundaryRelationHead(feature_size, 8)

    def encode_pacer_global(
        self, global_input: torch.Tensor, compute_occupancy: bool = True
    ):
        if not self.enable_pacer:
            raise RuntimeError("PACER is disabled for this network")
        return self.pacer_module.encode_global(
            global_input, compute_occupancy=compute_occupancy
        )

    def proj_feat(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.view(
            tensor.size(0), *self.feat_size, self.hidden_size
        )
        return tensor.permute(0, 4, 1, 2, 3).contiguous()

    def _apply_pacer(
        self,
        dec1: torch.Tensor,
        global_input: torch.Tensor,
        patch_center: torch.Tensor,
        case_shape: torch.Tensor,
        use_pacer: bool,
    ):
        if not self.enable_pacer or not use_pacer:
            return dec1, None
        if global_input is None or patch_center is None or case_shape is None:
            raise RuntimeError("PACER training requires whole-case context and patch geometry")
        memory, occupancy = self.encode_pacer_global(global_input)
        aligned, context = self.pacer_module.align_local(
            dec1, memory, patch_center, case_shape
        )
        context.update(
            {
                "memory": memory,
                "occupancy_logits": occupancy,
                "dec1_base": dec1,
                "dec1_aligned": aligned,
            }
        )
        return aligned, context

    def _forward_impl(
        self,
        x_in: torch.Tensor,
        return_training_aux: bool,
        global_input: torch.Tensor = None,
        patch_center: torch.Tensor = None,
        case_shape: torch.Tensor = None,
        use_pacer: bool = False,
    ):
        _, hidden_states = self.glicore_encoder(x_in)
        conv_block = self.encoder1(x_in)
        enc1, enc2, enc3, enc4 = hidden_states
        dec4 = self.proj_feat(enc4)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1_base = self.decoder3(dec2, enc1)
        dec1, pacer_aux = self._apply_pacer(
            dec1_base, global_input, patch_center, case_shape, use_pacer
        )
        out = self.decoder2(dec1, conv_block)

        logits_3 = self.out3(dec2)
        temp_3 = (F.softplus(self.temp_out3(dec2)) + 1e-2).clamp(max=5.0)
        evidence_3 = F.softplus(logits_3 / temp_3) + 1e-4
        logits_2 = self.out2(dec1)
        temp_2 = (F.softplus(self.temp_out2(dec1)) + 1e-2).clamp(max=5.0)
        evidence_2 = F.softplus(logits_2 / temp_2) + 1e-4

        out = self.faec_module(out)
        out = self.fece_module(out, evidence_3, evidence_2)
        out = self.uber_module(out)
        out = self.adater_module(out)
        logits_1 = self.out1(out)
        temp_1 = (F.softplus(self.temp_out1(out)) + 1e-2).clamp(max=5.0)
        evidence_1 = F.softplus(logits_1 / temp_1) + 1e-4
        output = [evidence_1, evidence_2, evidence_3] if self.do_ds else evidence_1

        if not return_training_aux:
            return output
        aux = {
            "feature": out,
            "logits_1": logits_1,
            "temperature_1": temp_1,
            "evidence_1": evidence_1,
        }
        if self.enable_pacer and pacer_aux is not None:
            aux.update(
                {
                    "pacer_memory": pacer_aux["memory"],
                    "pacer_occupancy_logits": pacer_aux["occupancy_logits"],
                    "pacer_relation_embedding": self.pacer_relation_head(out),
                    "pacer_context_residual": pacer_aux["context_residual"],
                    "pacer_dec1_base": pacer_aux["dec1_base"],
                    "pacer_dec1_aligned": pacer_aux["dec1_aligned"],
                }
            )
        return output, aux

    def forward(self, x: torch.Tensor):
        return self._forward_impl(
            x, False, use_pacer=False
        )

    def forward_training_aux(
        self,
        x_in: torch.Tensor,
        global_input: torch.Tensor,
        patch_center: torch.Tensor,
        case_shape: torch.Tensor,
    ):
        if not self.training:
            raise RuntimeError("forward_training_aux is available only in training mode")
        return self._forward_impl(
            x_in,
            True,
            global_input=global_input,
            patch_center=patch_center,
            case_shape=case_shape,
            use_pacer=global_input is not None,
        )
