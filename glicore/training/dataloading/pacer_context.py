import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _context_signature(
    modalities: int,
    spacing: Sequence[float],
    output_size: int,
    memory_size: int,
    num_classes: int,
) -> str:
    payload = {
        "version": 1,
        "modalities": int(modalities),
        "spacing": [float(value) for value in spacing],
        "output_size": int(output_size),
        "memory_size": int(memory_size),
        "num_classes": int(num_classes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()


def _fit_shape_to_cube(
    shape: Sequence[int], spacing: Sequence[float], output_size: int
):
    physical_extent = np.asarray(shape, dtype=np.float64) * np.asarray(
        spacing, dtype=np.float64
    )
    scale = float(output_size) / max(float(physical_extent.max()), 1e-6)
    resized_shape = np.maximum(
        1, np.rint(physical_extent * scale).astype(np.int64)
    )
    resized_shape = np.minimum(resized_shape, output_size)
    lower = (output_size - resized_shape) // 2
    upper = output_size - resized_shape - lower
    return resized_shape, lower, upper


def _build_occupancy_from_original(
    label: np.ndarray,
    resized_shape: np.ndarray,
    lower_padding: np.ndarray,
    output_size: int,
    memory_size: int,
    num_classes: int,
) -> np.ndarray:
    occupancy = np.zeros(
        (num_classes - 1, memory_size, memory_size, memory_size), dtype=np.uint8
    )
    original_shape = np.asarray(label.shape, dtype=np.float64)
    for class_index in range(1, num_classes):
        coordinates = np.argwhere(label == class_index)
        if coordinates.size == 0:
            continue
        scaled = np.floor(
            (coordinates.astype(np.float64) + 0.5)
            * resized_shape[None]
            / original_shape[None]
        ).astype(np.int64)
        scaled = np.minimum(scaled, resized_shape[None] - 1)
        global_coordinates = scaled + lower_padding[None]
        memory_coordinates = np.floor(
            global_coordinates.astype(np.float64)
            * memory_size
            / float(output_size)
        ).astype(np.int64)
        memory_coordinates = np.clip(memory_coordinates, 0, memory_size - 1)
        occupancy[
            class_index - 1,
            memory_coordinates[:, 0],
            memory_coordinates[:, 1],
            memory_coordinates[:, 2],
        ] = 1
    return occupancy


def build_pacer_case_context(
    case_data: np.ndarray,
    spacing,
    num_classes: int,
    output_size: int = 64,
    memory_size: int = 8,
) -> dict:
    if case_data.ndim != 4 or case_data.shape[0] < 2:
        raise ValueError("case_data must have shape [modalities + label, D, H, W]")
    image = np.asarray(case_data[:-1], dtype=np.float32)
    label = np.asarray(case_data[-1], dtype=np.int64)
    case_shape = np.asarray(label.shape, dtype=np.int32)
    resized_shape, lower, upper = _fit_shape_to_cube(
        case_shape, spacing, output_size
    )

    image_tensor = torch.from_numpy(image).unsqueeze(0)
    resized_image = F.interpolate(
        image_tensor,
        size=tuple(int(value) for value in resized_shape),
        mode="trilinear",
        align_corners=False,
    )[0]
    resized_label = F.interpolate(
        torch.from_numpy(label.astype(np.float32))[None, None],
        size=tuple(int(value) for value in resized_shape),
        mode="nearest",
    )[0, 0].to(torch.int16)
    padding = (
        int(lower[2]),
        int(upper[2]),
        int(lower[1]),
        int(upper[1]),
        int(lower[0]),
        int(upper[0]),
    )
    padded_image = F.pad(resized_image, padding)
    padded_label = F.pad(resized_label, padding, value=0)
    valid_mask = F.pad(
        torch.ones(tuple(int(value) for value in resized_shape), dtype=torch.uint8),
        padding,
        value=0,
    )
    occupancy = _build_occupancy_from_original(
        label,
        resized_shape,
        lower,
        output_size,
        memory_size,
        num_classes,
    )
    global_input = torch.cat(
        [padded_image, valid_mask.unsqueeze(0).to(padded_image.dtype)], dim=0
    )
    return {
        "image": padded_image.numpy().astype(np.float16),
        "global_input": global_input.numpy().astype(np.float32),
        "valid_mask": valid_mask.numpy().astype(np.uint8),
        "label": padded_label.numpy().astype(np.int16),
        "occupancy": occupancy,
        "case_shape": case_shape,
        "resized_shape": resized_shape.astype(np.int32),
        "lower_padding": lower.astype(np.int32),
    }


class PACERContextStore:
    def __init__(
        self,
        cache_dir,
        dataset,
        spacing,
        num_classes: int,
        lru_size: int = 64,
        output_size: int = 64,
        memory_size: int = 8,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.spacing = tuple(float(value) for value in spacing)
        self.num_classes = int(num_classes)
        self.lru_size = int(lru_size)
        self.output_size = int(output_size)
        self.memory_size = int(memory_size)
        self._cache = OrderedDict()

        first = next(iter(dataset.values()))
        case_data = self._load_case_data(first)
        self.signature = _context_signature(
            case_data.shape[0] - 1,
            self.spacing,
            self.output_size,
            self.memory_size,
            self.num_classes,
        )

    @staticmethod
    def _load_case_data(case_entry) -> np.ndarray:
        data_file = case_entry["data_file"]
        unpacked = data_file[:-4] + ".npy"
        if Path(unpacked).is_file():
            return np.load(unpacked, mmap_mode="r")
        with np.load(data_file) as archive:
            return archive["data"]

    @staticmethod
    def _safe_key(key: str) -> str:
        return str(key).replace("/", "_").replace("\\", "_")

    def _path(self, key: str) -> Path:
        return self.cache_dir / (self._safe_key(key) + ".npz")

    def prepare_case(self, key: str, overwrite: bool = False) -> str:
        path = self._path(key)
        if path.is_file() and not overwrite:
            try:
                with np.load(str(path), allow_pickle=False) as archive:
                    signature = str(archive["signature"].item())
                if signature == self.signature:
                    return "reused"
            except (KeyError, ValueError, OSError):
                pass

        case_data = self._load_case_data(self.dataset[key])
        context = build_pacer_case_context(
            case_data,
            spacing=self.spacing,
            num_classes=self.num_classes,
            output_size=self.output_size,
            memory_size=self.memory_size,
        )
        temporary_path = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            str(temporary_path),
            image=context["image"],
            valid_mask=context["valid_mask"],
            occupancy=context["occupancy"],
            case_shape=context["case_shape"],
            resized_shape=context["resized_shape"],
            lower_padding=context["lower_padding"],
            signature=np.asarray(self.signature),
        )
        temporary_path.replace(path)
        return "created"

    def require_cases(self, keys: Iterable[str]):
        for key in keys:
            path = self._path(key)
            if not path.is_file():
                raise RuntimeError(
                    "Missing PACER context for case '%s'. Run cache preparation first."
                    % key
                )
            try:
                with np.load(str(path), allow_pickle=False) as archive:
                    signature = str(archive["signature"].item())
            except (KeyError, ValueError, OSError) as error:
                raise RuntimeError("Invalid PACER context for case '%s'" % key) from error
            if signature != self.signature:
                raise RuntimeError(
                    "PACER context signature mismatch for case '%s'. Rebuild the cache."
                    % key
                )

    def _get(self, key: str) -> dict:
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        path = self._path(key)
        with np.load(str(path), allow_pickle=False) as archive:
            if str(archive["signature"].item()) != self.signature:
                raise RuntimeError("PACER context signature mismatch for case '%s'" % key)
            value = {
                "image": np.ascontiguousarray(archive["image"]),
                "valid_mask": np.ascontiguousarray(archive["valid_mask"]),
                "occupancy": np.ascontiguousarray(archive["occupancy"]),
                "case_shape": np.ascontiguousarray(archive["case_shape"]),
            }
        self._cache[key] = value
        while len(self._cache) > self.lru_size:
            self._cache.popitem(last=False)
        return value

    def get_batch(self, keys) -> dict:
        keys = [str(key) for key in keys]
        self.require_cases(keys)
        cases = [self._get(key) for key in keys]
        return {
            name: np.ascontiguousarray(np.stack([case[name] for case in cases]))
            for name in ("image", "valid_mask", "occupancy", "case_shape")
        }

    def compute_pos_weight(self, keys, maximum: float = 10.0) -> np.ndarray:
        keys = [str(key) for key in keys]
        self.require_cases(keys)
        positive = np.zeros(self.num_classes - 1, dtype=np.float64)
        valid_total = 0
        for key in keys:
            case = self._get(key)
            occupancy = case["occupancy"]
            valid = F.adaptive_max_pool3d(
                torch.from_numpy(case["valid_mask"])[None, None].float(),
                self.memory_size,
            )[0, 0].numpy() > 0.5
            positive += (
                occupancy[:, valid].reshape(occupancy.shape[0], -1).sum(axis=1)
            )
            valid_total += int(valid.sum())
        negative = np.maximum(float(valid_total) - positive, 0.0)
        return np.clip(negative / np.maximum(positive, 1.0), 1.0, maximum).astype(
            np.float32
        )
