from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover - dependency guard
    torch = None
    nn = None
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover - dependency guard
    TORCH_IMPORT_ERROR = None

from utils.dataset import blocks_to_image, clip_image, crop_to_original_shape, image_to_blocks, pad_image_to_block_size

BLOCK_SIZE = 128
N_OUTPUT = BLOCK_SIZE * BLOCK_SIZE
SAMPLING_BLOCK_SIZE = 32
CS_RATIO_TO_MEASUREMENT = {10: 102, 25: 256, 40: 410, 50: 512}


@dataclass
class PreparedReferenceModel:
    network: Any
    device: Any
    checkpoint_path: Path
    cs_ratio: int
    epoch_num: int
    layer_num: int
    learning_rate: float
    group_num: int


def dependencies_available() -> bool:
    return torch is not None


def require_dependencies() -> None:
    if torch is None:
        raise RuntimeError(f"PyTorch is required for this backend: {TORCH_IMPORT_ERROR}")


def sampling_rate_to_cs_ratio(sampling_rate: float | int) -> int:
    numeric = float(sampling_rate)
    if numeric > 1.0:
        candidate = int(round(numeric))
    else:
        candidate = int(round(numeric * 100))

    if candidate not in CS_RATIO_TO_MEASUREMENT:
        supported = ", ".join(str(item) for item in sorted(CS_RATIO_TO_MEASUREMENT))
        raise ValueError(
            f"Unsupported sampling rate {sampling_rate}. Supported CS ratios: {supported}."
        )
    return candidate


def measurement_count_from_ratio(cs_ratio: int) -> int:
    try:
        return CS_RATIO_TO_MEASUREMENT[cs_ratio]
    except KeyError as exc:
        supported = ", ".join(str(item) for item in sorted(CS_RATIO_TO_MEASUREMENT))
        raise ValueError(f"Unsupported cs_ratio={cs_ratio}. Supported: {supported}") from exc


def resolve_device(device: str) -> Any:
    require_dependencies()
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def expected_model_dir_name(
    model_prefix: str,
    layer_num: int,
    group_num: int,
    cs_ratio: int,
    learning_rate: float,
) -> str:
    return (
        f"{model_prefix}_layer_{layer_num}_group_{group_num}_ratio_{cs_ratio}_"
        f"lr_{learning_rate:.4f}"
    )


def resolve_checkpoint_file(
    checkpoint_dir: str | Path,
    model_prefix: str,
    layer_num: int,
    group_num: int,
    cs_ratio: int,
    learning_rate: float,
    epoch_num: int,
) -> Path:
    root = Path(checkpoint_dir)
    if root.is_file():
        return root

    if not root.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {root}")

    target_name = f"net_params_{epoch_num}.pkl"
    expected_dir = expected_model_dir_name(
        model_prefix=model_prefix,
        layer_num=layer_num,
        group_num=group_num,
        cs_ratio=cs_ratio,
        learning_rate=learning_rate,
    )

    preferred_paths = []
    if root.name == expected_dir:
        preferred_paths.append(root / target_name)
    else:
        preferred_paths.append(root / expected_dir / target_name)
        preferred_paths.append(root / target_name)

    for path in preferred_paths:
        if path.exists():
            return path

    candidates = sorted(root.rglob(target_name))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {target_name} under {root} for expected model directory {expected_dir}"
        )

    scored = sorted(
        candidates,
        key=lambda path: (
            0 if expected_dir in str(path.parent) else 1,
            len(path.parts),
            str(path),
        ),
    )
    return scored[0]


def prepare_block_input(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    padded, _ = pad_image_to_block_size(image, BLOCK_SIZE)
    blocks = image_to_blocks(padded, BLOCK_SIZE)
    return blocks, image.shape[:2], padded.shape


def restore_block_output(
    block_predictions: np.ndarray,
    padded_shape: tuple[int, int],
    original_shape: tuple[int, int],
) -> np.ndarray:
    padded = blocks_to_image(block_predictions, padded_shape, BLOCK_SIZE)
    cropped = crop_to_original_shape(padded, original_shape)
    return clip_image(cropped)


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        normalized[new_key] = value
    return normalized


def load_model_weights(model: Any, checkpoint_path: str | Path, device: Any) -> None:
    require_dependencies()
    payload = torch.load(Path(checkpoint_path), map_location=device)
    if isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            payload = payload["state_dict"]
        elif "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            payload = payload["model_state_dict"]

    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")

    state_dict = normalize_state_dict_keys(payload)
    model.load_state_dict(state_dict, strict=True)
