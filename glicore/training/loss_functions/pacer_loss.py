from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F


def pacer_occupancy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    target = target.to(device=logits.device, dtype=logits.dtype)
    valid_mask = valid_mask.to(device=logits.device, dtype=logits.dtype)
    if valid_mask.ndim == 4:
        valid_mask = valid_mask.unsqueeze(1)
    pos_weight = pos_weight.to(device=logits.device, dtype=logits.dtype)
    positive_weight = pos_weight.view(1, -1, 1, 1, 1)
    element_weight = torch.where(target > 0.5, positive_weight, 1.0)
    loss = F.binary_cross_entropy_with_logits(
        logits, target, weight=element_weight, reduction="none"
    )
    weighted = loss * valid_mask
    denominator = valid_mask.sum().clamp_min(1.0) * logits.shape[1]
    return weighted.sum() / denominator


def _balanced_coordinate_sample(
    mask: torch.Tensor,
    first_labels: torch.Tensor,
    second_labels: torch.Tensor,
    maximum: int,
) -> torch.Tensor:
    coordinates = torch.nonzero(mask, as_tuple=False)
    if coordinates.shape[0] <= maximum:
        return coordinates
    labels_a = first_labels[tuple(coordinates.t())]
    labels_b = second_labels[tuple(coordinates.t())]
    low = torch.minimum(labels_a, labels_b)
    high = torch.maximum(labels_a, labels_b)
    base = int(torch.max(high).item()) + 1
    interface = low * base + high
    unique_interfaces = torch.unique(interface)
    per_interface = max(1, maximum // max(1, unique_interfaces.numel()))
    selected = []
    for value in unique_interfaces:
        indices = torch.nonzero(interface == value, as_tuple=False).flatten()
        if indices.numel() > per_interface:
            indices = indices[
                torch.randperm(indices.numel(), device=indices.device)[:per_interface]
            ]
        selected.append(indices)
    selected = torch.cat(selected)
    if selected.numel() > maximum:
        selected = selected[
            torch.randperm(selected.numel(), device=selected.device)[:maximum]
        ]
    return coordinates[selected]


def _gather_pair_embeddings(
    embedding: torch.Tensor, coordinates: torch.Tensor, axis: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if coordinates.numel() == 0:
        empty = embedding.new_empty((0, embedding.shape[1]))
        return empty, empty
    second = coordinates.clone()
    second[:, axis + 1] += 1
    first_embedding = embedding[
        coordinates[:, 0],
        :,
        coordinates[:, 1],
        coordinates[:, 2],
        coordinates[:, 3],
    ]
    second_embedding = embedding[
        second[:, 0],
        :,
        second[:, 1],
        second[:, 2],
        second[:, 3],
    ]
    return first_embedding, second_embedding


def _interface_ids(
    coordinates: torch.Tensor,
    first_labels: torch.Tensor,
    second_labels: torch.Tensor,
    base: int,
) -> torch.Tensor:
    labels_a = first_labels[tuple(coordinates.t())]
    labels_b = second_labels[tuple(coordinates.t())]
    low = torch.minimum(labels_a, labels_b)
    high = torch.maximum(labels_a, labels_b)
    return low * base + high


def _mean_over_interfaces(
    values: torch.Tensor, interfaces: torch.Tensor
) -> torch.Tensor:
    return torch.stack(
        [values[interfaces == value].mean() for value in torch.unique(interfaces)]
    ).mean()


def _balanced_index_sample(
    interfaces: torch.Tensor, maximum: int
) -> torch.Tensor:
    if interfaces.shape[0] <= maximum:
        return torch.arange(interfaces.shape[0], device=interfaces.device)
    unique_interfaces = torch.unique(interfaces)
    per_interface = max(1, maximum // max(1, unique_interfaces.numel()))
    selected = []
    for value in unique_interfaces:
        indices = torch.nonzero(
            interfaces == value, as_tuple=False
        ).flatten()
        if indices.numel() > per_interface:
            indices = indices[
                torch.randperm(indices.numel(), device=indices.device)[
                    :per_interface
                ]
            ]
        selected.append(indices)
    selected = torch.cat(selected)
    if selected.numel() > maximum:
        selected = selected[
            torch.randperm(selected.numel(), device=selected.device)[:maximum]
        ]
    return selected


def pacer_relation_loss(
    embedding: torch.Tensor,
    target: torch.Tensor,
    spacing: Sequence[float],
    max_pairs: int = 2048,
    margin: float = 0.3,
    valid_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if target.ndim == 4:
        target = target.unsqueeze(1)
    if target.shape[2:] != embedding.shape[2:]:
        target = F.interpolate(target.float(), size=embedding.shape[2:], mode="nearest")
    labels = target[:, 0].long()
    valid = labels >= 0
    if valid_mask is not None:
        if valid_mask.ndim == 4:
            valid_mask = valid_mask.unsqueeze(1)
        if valid_mask.shape[2:] != embedding.shape[2:]:
            valid_mask = F.interpolate(
                valid_mask.float(), size=embedding.shape[2:], mode="nearest"
            )
        valid = valid & (valid_mask[:, 0] > 0.5)
    interface_base = int(labels[valid].max().item()) + 1 if valid.any() else 1

    label_float = labels.clamp_min(0).float()
    max_input = torch.where(valid, label_float, label_float.new_full((), -1.0))
    min_input = torch.where(
        valid,
        label_float,
        label_float.new_full((), float(interface_base)),
    )
    local_max = F.max_pool3d(
        max_input.unsqueeze(1), kernel_size=3, stride=1, padding=1
    )
    local_min = -F.max_pool3d(
        -min_input.unsqueeze(1), kernel_size=3, stride=1, padding=1
    )
    boundary = (local_max != local_min)[:, 0] & valid

    pull_first = []
    pull_second = []
    pull_interfaces = []
    push_first = []
    push_second = []
    push_interfaces = []
    pairs_per_axis = max(1, max_pairs // 3)

    for axis in range(3):
        if float(spacing[axis]) > 3.0:
            continue
        left = [slice(None)] * 4
        right = [slice(None)] * 4
        left[axis + 1] = slice(0, -1)
        right[axis + 1] = slice(1, None)
        label_left = labels[tuple(left)]
        label_right = labels[tuple(right)]
        valid_pair = valid[tuple(left)] & valid[tuple(right)]
        boundary_pair = boundary[tuple(left)] | boundary[tuple(right)]

        pull_mask = (
            valid_pair
            & boundary_pair
            & (label_left == label_right)
            & (label_left > 0)
        )
        push_mask = (
            valid_pair
            & (label_left != label_right)
            & ((label_left > 0) | (label_right > 0))
        )
        pull_coordinates = _balanced_coordinate_sample(
            pull_mask, label_left, label_right, pairs_per_axis
        )
        push_coordinates = _balanced_coordinate_sample(
            push_mask, label_left, label_right, pairs_per_axis
        )
        first, second = _gather_pair_embeddings(
            embedding, pull_coordinates, axis
        )
        if first.numel():
            pull_first.append(first)
            pull_second.append(second)
            pull_interfaces.append(
                _interface_ids(
                    pull_coordinates, label_left, label_right, interface_base
                )
            )
        first, second = _gather_pair_embeddings(
            embedding, push_coordinates, axis
        )
        if first.numel():
            push_first.append(first)
            push_second.append(second)
            push_interfaces.append(
                _interface_ids(
                    push_coordinates, label_left, label_right, interface_base
                )
            )

    zero = embedding.sum() * 0.0
    pull_loss = zero
    push_loss = zero
    pull_count = 0
    push_count = 0
    mean_pull = 0.0
    mean_push = 0.0
    pull_budget = max_pairs
    push_budget = max_pairs
    if pull_first and push_first:
        pull_budget = max(1, max_pairs // 2)
        push_budget = max(1, max_pairs - pull_budget)

    if pull_first:
        first = torch.cat(pull_first)
        second = torch.cat(pull_second)
        interfaces = torch.cat(pull_interfaces)
        if first.shape[0] > pull_budget:
            selected = _balanced_index_sample(interfaces, pull_budget)
            first = first[selected]
            second = second[selected]
            interfaces = interfaces[selected]
        pull_distance = 1.0 - F.cosine_similarity(first, second, dim=1)
        pull_loss = _mean_over_interfaces(pull_distance, interfaces)
        pull_count = first.shape[0]
        mean_pull = float(
            _mean_over_interfaces(pull_distance.detach(), interfaces).cpu()
        )

    if push_first:
        first = torch.cat(push_first)
        second = torch.cat(push_second)
        interfaces = torch.cat(push_interfaces)
        if first.shape[0] > push_budget:
            selected = _balanced_index_sample(interfaces, push_budget)
            first = first[selected]
            second = second[selected]
            interfaces = interfaces[selected]
        push_similarity = F.cosine_similarity(first, second, dim=1)
        push_distance = 1.0 - push_similarity
        push_loss = _mean_over_interfaces(
            F.relu(margin - push_distance), interfaces
        )
        push_count = first.shape[0]
        mean_push = float(
            _mean_over_interfaces(push_distance.detach(), interfaces).cpu()
        )

    return pull_loss + push_loss, {
        "pull_pairs": pull_count,
        "push_pairs": push_count,
        "mean_pull_distance": mean_pull,
        "mean_push_distance": mean_push,
    }
