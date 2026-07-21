from __future__ import annotations

from typing import Any

import numpy as np

try:
    from skimage.metrics import structural_similarity as skimage_structural_similarity
except ImportError:  # pragma: no cover - optional dependency fallback
    skimage_structural_similarity = None


FULL_Y_DATA_RANGE = 1.0


def _as_float_image(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def compute_psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float = FULL_Y_DATA_RANGE,
) -> float:
    prediction = _as_float_image(prediction)
    target = _as_float_image(target)
    mse = float(np.mean((prediction - target) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def _global_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float = FULL_Y_DATA_RANGE,
) -> float:
    prediction = _as_float_image(prediction)
    target = _as_float_image(target)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = float(prediction.mean())
    mu_y = float(target.mean())
    sigma_x = float(prediction.var())
    sigma_y = float(target.var())
    sigma_xy = float(((prediction - mu_x) * (target - mu_y)).mean())

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return float(numerator / denominator)


def compute_ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float = FULL_Y_DATA_RANGE,
) -> float:
    prediction = _as_float_image(prediction)
    target = _as_float_image(target)

    if prediction.shape != target.shape:
        raise ValueError(
            f"SSIM expects images of the same shape, got {prediction.shape} and {target.shape}"
        )

    if skimage_structural_similarity is None:
        return _global_ssim(prediction, target, data_range=data_range)

    min_side = min(prediction.shape[0], prediction.shape[1])
    if min_side < 7:
        return _global_ssim(prediction, target, data_range=data_range)

    kwargs: dict[str, Any] = {"data_range": data_range, "win_size": 7}
    if prediction.ndim == 3:
        kwargs["channel_axis"] = -1

    return float(skimage_structural_similarity(target, prediction, **kwargs))


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    data_range: float = FULL_Y_DATA_RANGE,
) -> dict[str, float]:
    return {
        "psnr": compute_psnr(prediction, target, data_range=data_range),
        "ssim": compute_ssim(prediction, target, data_range=data_range),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise ValueError("Cannot summarize an empty record list.")

    psnr_values = [float(record["psnr"]) for record in records]
    ssim_values = [float(record["ssim"]) for record in records]
    time_values = [float(record["time_sec"]) for record in records]

    return {
        "avg_psnr": _safe_mean(psnr_values),
        "avg_ssim": _safe_mean(ssim_values),
        "avg_time_sec": _safe_mean(time_values),
    }


def _safe_mean(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("inf")
    return float(np.mean(finite_values))
