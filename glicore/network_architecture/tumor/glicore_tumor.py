from torch import nn
from typing import Tuple, Union
import torch
from glicore.network_architecture.neural_network import SegmentationNetwork
from glicore.network_architecture.dynunet_block import UnetResBlock
from glicore.network_architecture.tumor.model_components import GliCoReEncoder, GliCoReUpBlock
from glicore.network_architecture.tumor.mefc import EvidentialPredictionHead, MEFC
from glicore.network_architecture.tumor.pacer import (
    BoundaryRelationHead,
    PACER,
)
from glicore.network_architecture.tumor.ubtr import UBTR


class GliCoRe(SegmentationNetwork):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,
            enable_pacer: bool = False,
            pacer_patch_size=(128, 128, 128),
    ) -> None:
        super().__init__()
        if depths is None: depths = [3, 3, 3, 3]
        if dims is None: dims = [32, 64, 128, 256]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        self.hidden_size = hidden_size
        self.feat_size = (4, 4, 4,)
        self.enable_pacer = bool(enable_pacer)
        self.mask_policy = "et_wt"

        self.glicore_encoder = GliCoReEncoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder1 = UnetResBlock(spatial_dims=3, in_channels=in_channels, out_channels=feature_size, kernel_size=3,
                                     stride=1, norm_name=norm_name)
        self.decoder5 = GliCoReUpBlock(spatial_dims=3, in_channels=feature_size * 16, out_channels=feature_size * 8,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, out_size=8 * 8 * 8)
        self.decoder4 = GliCoReUpBlock(spatial_dims=3, in_channels=feature_size * 8, out_channels=feature_size * 4,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, out_size=16 * 16 * 16)
        self.decoder3 = GliCoReUpBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=feature_size * 2,
                                     kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, out_size=32 * 32 * 32)
        self.decoder2 = GliCoReUpBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=feature_size,
                                     kernel_size=3, upsample_kernel_size=(4, 4, 4), norm_name=norm_name,
                                     out_size=128 * 128 * 128, conv_decoder=True)

        self.mefc = MEFC(
            channels=feature_size,
            num_classes=out_channels,
            dec1_channels=feature_size * 2,
            dec2_channels=feature_size * 4,
            patch_size=8,
            full_resolution_size=128,
            target_class_index=min(3, out_channels - 1),
        )
        self.ubtr = UBTR(in_channels=feature_size, num_classes=out_channels)
        self.evidence_head_e1 = EvidentialPredictionHead(
            in_channels=feature_size,
            num_classes=out_channels,
        )
        if self.enable_pacer:
            self.pacer = PACER(
                image_channels=in_channels,
                local_channels=feature_size * 2,
                num_classes=out_channels,
                nominal_patch_size=pacer_patch_size,
            )
            self.pacer_relation_head = BoundaryRelationHead(
                in_channels=feature_size,
                embedding_channels=8,
            )


    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def encode_pacer_global(self, global_input, compute_occupancy=True):
        if not self.enable_pacer:
            raise RuntimeError("PACER is disabled for this network")
        return self.pacer.encode_global(
            global_input, compute_occupancy=compute_occupancy
        )

    def _apply_pacer(
        self,
        dec1_base,
        global_input,
        patch_center,
        case_shape,
        use_pacer,
    ):
        if not self.enable_pacer or not use_pacer:
            return dec1_base, {}
        if global_input is None or patch_center is None or case_shape is None:
            raise RuntimeError("PACER training requires whole-case context and patch geometry")
        memory, occupancy_logits = self.encode_pacer_global(
            global_input, compute_occupancy=True
        )
        aligned, route_aux = self.pacer.align_local(
            dec1_base, memory, patch_center, case_shape
        )
        route_aux.update({
            "memory": memory,
            "occupancy_logits": occupancy_logits,
            "dec1_base": dec1_base,
            "dec1_aligned": aligned,
        })
        return aligned, route_aux

    def _forward_impl(
        self,
        x_in,
        return_training_aux=False,
        global_input=None,
        patch_center=None,
        case_shape=None,
        use_pacer=False,
    ):
        x_output, hidden_states = self.glicore_encoder(x_in)
        convBlock = self.encoder1(x_in)

        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1_base = self.decoder3(dec2, enc1)
        dec1, pacer_aux = self._apply_pacer(
            dec1_base,
            global_input,
            patch_center,
            case_shape,
            use_pacer,
        )

        out = self.decoder2(dec1, convBlock)

        out, evidence_2, evidence_3 = self.mefc(out, dec1, dec2)
        out = self.ubtr(out)
        evidence_1, logits_1, temp_1 = self.evidence_head_e1(out)

        if self.do_ds:
            output = [evidence_1, evidence_2, evidence_3]
        else:
            output = evidence_1
        if return_training_aux:
            aux = {
                "feature": out,
                "logits_1": logits_1,
                "temperature_1": temp_1,
                "evidence_1": evidence_1,
            }
            if self.enable_pacer:
                aux.update({
                    "pacer_memory": pacer_aux["memory"],
                    "pacer_occupancy_logits": pacer_aux["occupancy_logits"],
                    "pacer_relation_embedding": self.pacer_relation_head(out),
                    "pacer_context_residual": pacer_aux["context_residual"],
                    "pacer_dec1_base": pacer_aux["dec1_base"],
                    "pacer_dec1_aligned": pacer_aux["dec1_aligned"],
                })
            return output, aux
        return output

    def forward(self, x):
        return self._forward_impl(
            x,
            return_training_aux=False,
            use_pacer=False,
        )

    def forward_training_aux(
        self,
        x_in,
        global_input=None,
        patch_center=None,
        case_shape=None,
    ):
        if not self.training:
            raise RuntimeError("forward_training_aux is available only in training mode")
        return self._forward_impl(
            x_in,
            return_training_aux=True,
            global_input=global_input,
            patch_center=patch_center,
            case_shape=case_shape,
            use_pacer=global_input is not None,
        )
