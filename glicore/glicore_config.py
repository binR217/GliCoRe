"""Paper-aligned constants for the public GliCoRe reproduction release."""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class DatasetProfile:
    in_channels: int
    num_classes: int
    patch_size: Tuple[int, int, int]
    mask_policy: str


@dataclass(frozen=True)
class PaperTrainingConfig:
    epochs: int = 1000
    optimizer: str = "nesterov_sgd"
    momentum: float = 0.99
    weight_decay: float = 3e-5
    initial_lr: float = 1e-2
    poly_power: float = 0.9
    sliding_window_overlap: float = 0.5
    gaussian_weighting: bool = True


@dataclass(frozen=True)
class PACERConfig:
    calibration_batches: int = 16
    occupancy_gradient_ratio: float = 0.5
    relation_gradient_ratio: float = 0.2
    occupancy_weight_clip: Tuple[float, float] = (0.02, 0.20)
    relation_weight_clip: Tuple[float, float] = (1e-4, 0.10)
    relation_margin: float = 0.3
    max_adjacent_pairs: int = 2048
    max_axis_spacing_mm: float = 3.0
    ramp_epochs: int = 30


@dataclass(frozen=True)
class MEFCConfig:
    patch_side: int = 8
    low_frequency_alpha: float = 2.0
    calibration_beta: float = 0.1
    temperature_offset: float = 0.01
    temperature_max: float = 5.0
    evidence_offset: float = 1e-4


PAPER_TRAINING = PaperTrainingConfig()
PACER_CONFIG = PACERConfig()
MEFC_CONFIG = MEFCConfig()

DATASET_PROFILES: Dict[str, DatasetProfile] = {
    "brats": DatasetProfile(4, 4, (128, 128, 128), "et_wt"),
    "synapse": DatasetProfile(1, 9, (64, 128, 128), "foreground_interface"),
    "acdc": DatasetProfile(1, 4, (16, 160, 160), "foreground_interface"),
}
