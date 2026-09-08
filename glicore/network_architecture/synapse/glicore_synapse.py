from typing import Sequence, Tuple, Union

from torch import nn

from glicore.network_architecture.glicore_generalized import (
    GliCoReGeneralized,
)
from glicore.network_architecture.synapse.model_components import (
    GliCoReEncoder,
    GliCoReUpBlock,
)


class GliCoRe(GliCoReGeneralized):
    """Synapse geometry with the generalized GliCoRe+PACER decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] = (64, 128, 128),
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        depths=None,
        dims=None,
        conv_op=nn.Conv3d,
        do_ds: bool = True,
        enable_pacer: bool = False,
        pacer_patch_size=None,
    ):
        if tuple(int(value) for value in img_size) != (64, 128, 128):
            raise ValueError("Synapse GliCoRe expects patch size (64, 128, 128)")
        super().__init__(
            encoder_class=GliCoReEncoder,
            up_block_class=GliCoReUpBlock,
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feat_size=(4, 4, 4),
            decoder_sizes=((8, 8, 8), (16, 16, 16), (32, 32, 32)),
            final_upsample=(2, 4, 4),
            feature_size=feature_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            norm_name=norm_name,
            depths=depths,
            dims=dims,
            conv_op=conv_op,
            do_ds=do_ds,
            enable_pacer=enable_pacer,
            faec_patch_size=8,
        )
