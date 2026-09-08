import torch
import torch.nn.functional as F
from torch import nn

class GliCoReEvidentialLoss(nn.Module):
    """Dataset-agnostic form of the evidential loss used by GliCoRe."""

    def __init__(
        self,
        soft_dice_kwargs,
        num_classes: int,
        max_epochs: int = 1000,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        lambda_kl: float = 0.1,
        lambda_freq: float = 0.05,
        lambda_bound: float = 0.1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.max_epochs = int(max_epochs)
        self.lambda_dice = float(lambda_dice)
        self.lambda_ce = float(lambda_ce)
        self.lambda_kl = float(lambda_kl)
        self.lambda_freq = float(lambda_freq)
        self.lambda_bound = float(lambda_bound)
        self.current_epoch = 0
        dice_kwargs = dict(soft_dice_kwargs)
        self.batch_dice = bool(dice_kwargs.get("batch_dice", False))
        self.dice_do_bg = bool(dice_kwargs.get("do_bg", True))
        self.dice_smooth = float(dice_kwargs.get("smooth", 1.0))
        self.register_buffer(
            "class_weights", torch.ones(self.num_classes), persistent=False
        )

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        weights = torch.as_tensor(class_weights, dtype=torch.float32).flatten()
        if weights.numel() != self.num_classes:
            raise ValueError(
                "class_weights must contain %d entries" % self.num_classes
            )
        if not torch.isfinite(weights).all() or torch.any(weights <= 0.0):
            raise ValueError("class_weights must be finite and strictly positive")
        self.class_weights.copy_(weights.to(self.class_weights.device))

    def _weighted_dice(
        self, probability: torch.Tensor, target_one_hot: torch.Tensor
    ) -> torch.Tensor:
        axes = [0] + list(range(2, probability.ndim)) if self.batch_dice else list(
            range(2, probability.ndim)
        )
        tp = (probability * target_one_hot).sum(dim=axes)
        fp = (probability * (1.0 - target_one_hot)).sum(dim=axes)
        fn = ((1.0 - probability) * target_one_hot).sum(dim=axes)
        dice = (2.0 * tp + self.dice_smooth) / (
            2.0 * tp + fp + fn + self.dice_smooth + 1e-8
        )
        weights = self.class_weights.to(probability)
        if not self.dice_do_bg:
            dice = dice[..., 1:]
            weights = weights[1:]
        return -((dice * weights).sum(dim=-1) / weights.sum()).mean()

    def _weighted_channel_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        reduce_axes = [0] + list(range(2, tensor.ndim))
        per_class = tensor.mean(dim=reduce_axes)
        weights = self.class_weights.to(tensor)
        return (per_class * weights).sum() / weights.sum()

    @staticmethod
    def _edges(tensor: torch.Tensor) -> torch.Tensor:
        padded = F.pad(tensor, (1, 1, 1, 1, 1, 1), mode="replicate")
        maximum = F.max_pool3d(padded, kernel_size=3, stride=1)
        minimum = -F.max_pool3d(-padded, kernel_size=3, stride=1)
        return maximum - minimum

    def forward(self, evidence: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 4:
            target = target.unsqueeze(1)
        target = target.long()
        target_one_hot = torch.zeros_like(evidence).scatter_(1, target, 1)

        alpha = evidence + 1.0
        strength = alpha.sum(dim=1, keepdim=True)
        probability = alpha / strength
        loss_dice = self._weighted_dice(probability, target_one_hot)
        ce_map = (
            target_one_hot * (torch.digamma(strength) - torch.digamma(alpha))
        ).sum(dim=1, keepdim=True)
        voxel_weights = self.class_weights.to(evidence)[target]
        loss_ce = (ce_map * voxel_weights).sum() / voxel_weights.sum().clamp_min(1.0)

        alpha_tilde = target_one_hot + (1.0 - target_one_hot) * alpha
        strength_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        ones = torch.ones_like(alpha_tilde)
        kl = (
            torch.lgamma(strength_tilde)
            - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(torch.ones_like(strength_tilde) * self.num_classes)
        )
        kl = kl + (
            (alpha_tilde - 1.0)
            * (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde))
        ).sum(dim=1, keepdim=True)
        annealing = min(
            1.0, float(self.current_epoch) / (self.max_epochs * 0.5 + 1e-8)
        )

        prediction_frequency = torch.fft.rfftn(
            probability.float(), dim=(2, 3, 4), norm="ortho"
        )
        target_frequency = torch.fft.rfftn(
            target_one_hot.float(), dim=(2, 3, 4), norm="ortho"
        )
        loss_frequency = self._weighted_channel_mean(
            (prediction_frequency - target_frequency).abs()
        )

        loss_edge = self._weighted_channel_mean(
            (self._edges(probability) - self._edges(target_one_hot)).abs()
        )
        background_mask = 1.0 - target_one_hot
        distance_weight = background_mask
        for _ in range(3):
            distance_weight = F.avg_pool3d(
                distance_weight, kernel_size=5, stride=1, padding=2
            )
        loss_distance = self._weighted_channel_mean(
            probability * background_mask * distance_weight
        )
        loss_boundary = loss_edge + 10.0 * loss_distance

        loss_kl = (kl * voxel_weights).sum() / voxel_weights.sum().clamp_min(1.0)

        return (
            self.lambda_ce * loss_ce
            + self.lambda_dice * loss_dice
            + self.lambda_kl * annealing * loss_kl
            + self.lambda_freq * loss_frequency
            + self.lambda_bound * loss_boundary
        )
