"""Uncertainty-Boundary Topology Refinement (UBTR)."""

import torch
import torch.nn.functional as F
from torch import nn


class UncertaintyBoundaryEdgeRefinement(nn.Module):
    """Inject fused spatial/frequency edges only at uncertain locations."""

    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.probe_head = nn.Conv3d(in_channels, num_classes, kernel_size=1)
        self.spatial_edge_branch = nn.Sequential(
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
        self.frequency_weight = nn.Parameter(
            torch.ones(1, in_channels, 1, 1, 1)
        )
        self.edge_fusion_gate = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
            nn.Sigmoid(),
        )
        self.uncertainty_scale = nn.Parameter(torch.tensor([5.0]))
        self.uncertainty_threshold = nn.Parameter(torch.tensor([0.2]))
        self.injection_scale = nn.Parameter(torch.tensor([0.5]))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        probe_logits = self.probe_head(feature)
        dirichlet_alpha = F.softplus(probe_logits) + 1.0
        uncertainty = self.num_classes / torch.sum(
            dirichlet_alpha, dim=1, keepdim=True
        )
        uncertainty_mask = torch.sigmoid(
            self.uncertainty_scale
            * (uncertainty - self.uncertainty_threshold)
        )

        spatial_edge = self.spatial_edge_branch(feature)
        spectrum = torch.fft.rfftn(
            feature.float(), dim=(2, 3, 4), norm="ortho"
        )
        frequency_edge = torch.fft.irfftn(
            spectrum * self.frequency_weight,
            s=feature.shape[2:],
            dim=(2, 3, 4),
            norm="ortho",
        ).to(feature.dtype)
        fusion_gate = self.edge_fusion_gate(
            torch.cat([spatial_edge, frequency_edge], dim=1)
        )
        edge_response = fusion_gate * feature
        return feature + self.injection_scale * uncertainty_mask * edge_response


class AdaptiveTopologyEvidenceRectification(nn.Module):
    """Rectify semantic features with morphology and class relations."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.semantic_projection = nn.Sequential(
            nn.Conv3d(in_channels, num_classes, kernel_size=3, padding=1),
            nn.InstanceNorm3d(num_classes),
        )
        self.morphological_envelope = nn.MaxPool3d(
            kernel_size=5, stride=1, padding=2
        )
        self.class_relation_matrix = nn.Parameter(
            torch.eye(num_classes)
            + 0.05 * torch.randn(num_classes, num_classes)
        )
        self.feature_reconstruction = nn.Sequential(
            nn.Conv3d(num_classes + in_channels, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
            nn.GELU(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        semantic_logits = self.semantic_projection(feature)
        evidence = F.softplus(semantic_logits)
        belief = evidence / (evidence + 1.0)
        envelopes = self.morphological_envelope(belief)

        batch, classes, depth, height, width = envelopes.shape
        flattened = envelopes.view(batch, classes, -1)
        relation_weights = torch.sigmoid(self.class_relation_matrix)
        topology_gate = torch.einsum(
            "ij,bjn->bin", relation_weights, flattened
        ).view(batch, classes, depth, height, width)
        rectified_semantics = semantic_logits * topology_gate
        return self.feature_reconstruction(
            torch.cat([feature, rectified_semantics], dim=1)
        )


class UBTR(nn.Module):
    """Progressive UBER then AdaTER feature refinement."""

    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        self.uber = UncertaintyBoundaryEdgeRefinement(
            in_channels=in_channels,
            num_classes=num_classes,
        )
        self.adater = AdaptiveTopologyEvidenceRectification(
            in_channels=in_channels,
            num_classes=num_classes,
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.adater(self.uber(feature))


UBER = UncertaintyBoundaryEdgeRefinement
AdaTER = AdaptiveTopologyEvidenceRectification
