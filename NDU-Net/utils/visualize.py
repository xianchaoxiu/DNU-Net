from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    array = (array * 255.0).round().astype(np.uint8)
    return array


def _ensure_rgb(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return np.stack([array, array, array], axis=-1)
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    raise ValueError(f"Unsupported image shape for visualization: {array.shape}")


def save_image(image: np.ndarray, save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = _ensure_rgb(_to_uint8(image))
    Image.fromarray(array).save(path)


def build_error_map(target: np.ndarray, prediction: np.ndarray, amplify: float = 4.0) -> np.ndarray:
    target = np.asarray(target, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    error = np.abs(target - prediction)

    if error.ndim == 3:
        error = error.mean(axis=2)

    error = np.clip(error * amplify, 0.0, 1.0)
    red = error
    green = np.zeros_like(error)
    blue = np.zeros_like(error)
    return np.stack([red, green, blue], axis=-1)


def save_triptych(
    target: np.ndarray,
    prediction: np.ndarray,
    save_path: str | Path,
    spacer: int = 6,
    diff_amplify: float = 4.0,
) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    target_rgb = _ensure_rgb(_to_uint8(target))
    prediction_rgb = _ensure_rgb(_to_uint8(prediction))
    error_rgb = _to_uint8(build_error_map(target, prediction, amplify=diff_amplify))

    height = target_rgb.shape[0]
    spacer_block = np.full((height, spacer, 3), 255, dtype=np.uint8)
    canvas = np.concatenate(
        [target_rgb, spacer_block, prediction_rgb, spacer_block, error_rgb],
        axis=1,
    )
    Image.fromarray(canvas).save(path)
