from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image
from skimage import io

SUPPORTED_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".pgm",
}


def rate_to_tag(rate: float) -> str:
    return f"{int(round(rate * 100)):03d}"


def find_image_paths(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    paths = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and not any(part.startswith(".") for part in path.relative_to(root).parts)
            and not path.name.lower().startswith("demo_")
            and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    ]
    paths.sort()
    return paths


def _rgb_to_full_range_y_channel(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized RGB to full-range BT.601 luminance in [0, 1]."""
    rgb = np.asarray(rgb, dtype=np.float32)
    return (
        0.299 * rgb[..., 0]
        + 0.587 * rgb[..., 1]
        + 0.114 * rgb[..., 2]
    ).astype(np.float32)


def _rgb_to_full_range_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """Convert normalized RGB to full-range BT.601 Y'CbCr in [0, 1]."""
    rgb = np.asarray(rgb, dtype=np.float32)
    y_channel = _rgb_to_full_range_y_channel(rgb)
    cb_channel = (
        -0.168736 * rgb[..., 0]
        - 0.331264 * rgb[..., 1]
        + 0.5 * rgb[..., 2]
        + 0.5
    )
    cr_channel = (
        0.5 * rgb[..., 0]
        - 0.418688 * rgb[..., 1]
        - 0.081312 * rgb[..., 2]
        + 0.5
    )
    return np.stack([y_channel, cb_channel, cr_channel], axis=-1).astype(np.float32)


def _full_range_ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """Convert full-range BT.601 Y'CbCr in [0, 1] to normalized RGB."""
    ycbcr = np.asarray(ycbcr, dtype=np.float32)
    y_channel = ycbcr[..., 0]
    cb_offset = ycbcr[..., 1] - 0.5
    cr_offset = ycbcr[..., 2] - 0.5

    red = y_channel + 1.402 * cr_offset
    green = y_channel - 0.344136 * cb_offset - 0.714136 * cr_offset
    blue = y_channel + 1.772 * cb_offset
    rgb = np.stack([red, green, blue], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _normalize_image_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if np.issubdtype(array.dtype, np.integer):
        max_value = float(np.iinfo(array.dtype).max)
        return array.astype(np.float32) / max_value
    return array.astype(np.float32)


def _read_image_array(image_path: str | Path) -> np.ndarray:
    return np.asarray(io.imread(Path(image_path)))


def _read_rgb_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    array = _read_image_array(path)

    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3:
        return array[..., :3]
    raise ValueError(f"Unsupported image shape for {path}: {array.shape}")


def load_y_channel_image(image_path: str | Path) -> np.ndarray:
    """Load full-range luminance: normalize grayscale or convert normalized RGB."""
    path = Path(image_path)
    array = _read_image_array(path)
    if array.ndim == 2:
        return _normalize_image_array(array)
    if array.ndim == 3 and array.shape[2] == 1:
        return _normalize_image_array(array[..., 0])
    if array.ndim == 3 and array.shape[2] >= 3:
        return _rgb_to_full_range_y_channel(_normalize_image_array(array[..., :3]))
    raise ValueError(f"Unsupported image shape for {path}: {array.shape}")


def compose_rgb_from_y_channel(y_channel: np.ndarray, source_image_path: str | Path) -> np.ndarray:
    y_channel = np.asarray(y_channel, dtype=np.float32)
    if y_channel.ndim != 2:
        raise ValueError(f"Expected a Y-channel image with shape [H, W], got {y_channel.shape}")

    source_rgb = _normalize_image_array(_read_rgb_image(source_image_path))
    source_ycbcr = _rgb_to_full_range_ycbcr(source_rgb)
    if source_ycbcr.shape[:2] != y_channel.shape:
        raise ValueError(
            "Y-channel shape must match source image shape, got "
            f"{y_channel.shape} and {source_ycbcr.shape[:2]}"
        )

    composed_ycbcr = source_ycbcr.copy()
    composed_ycbcr[..., 0] = np.clip(y_channel, 0.0, 1.0)
    return _full_range_ycbcr_to_rgb(composed_ycbcr)


def load_image(image_path: str | Path, grayscale: bool = True) -> np.ndarray:
    if grayscale:
        return load_y_channel_image(image_path)
    return _normalize_image_array(_read_rgb_image(image_path))


def clip_image(image: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)


def compute_padding(height: int, width: int, block_size: int) -> tuple[int, int]:
    pad_h = (block_size - height % block_size) % block_size
    pad_w = (block_size - width % block_size) % block_size
    return pad_h, pad_w


def pad_image_to_block_size(image: np.ndarray, block_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    if image.ndim != 2:
        raise ValueError("Block-CS helpers currently expect Y-channel images with shape [H, W].")

    height, width = image.shape
    pad_h, pad_w = compute_padding(height, width, block_size)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, (pad_h, pad_w)


def crop_to_original_shape(image: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    height, width = original_shape
    return np.asarray(image, dtype=np.float32)[:height, :width]


def image_to_blocks(image: np.ndarray, block_size: int) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError("Expected a Y-channel image with shape [H, W].")

    height, width = image.shape
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError("Image shape must be divisible by block size before block extraction.")

    blocks = (
        image.reshape(height // block_size, block_size, width // block_size, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, block_size * block_size)
    )
    return np.asarray(blocks, dtype=np.float32)


def blocks_to_image(blocks: np.ndarray, image_shape: tuple[int, int], block_size: int) -> np.ndarray:
    height, width = image_shape
    expected_blocks = (height // block_size) * (width // block_size)
    if blocks.shape[0] != expected_blocks:
        raise ValueError(
            f"Block count mismatch: expected {expected_blocks}, got {blocks.shape[0]}"
        )

    image = (
        blocks.reshape(height // block_size, width // block_size, block_size, block_size)
        .transpose(0, 2, 1, 3)
        .reshape(height, width)
    )
    return np.asarray(image, dtype=np.float32)
