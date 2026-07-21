from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.ndu_net.train import MODEL_PREFIX, NDUNet
from methods.ndu_net.train import BEST_CHECKPOINT_NAME, DEFAULT_LAMBDA_NEWTON, METHOD_NAME, target_checkpoint_dir
from methods.ndu_net.train import effective_num_stages
from methods.cs_reference import (
    BLOCK_SIZE,
    SAMPLING_BLOCK_SIZE,
    PreparedReferenceModel,
    dependencies_available,
    load_model_weights,
    measurement_count_from_ratio,
    require_dependencies,
    resolve_checkpoint_file,
    resolve_device,
    sampling_rate_to_cs_ratio,
    torch,
)
from utils.dataset import clip_image, crop_to_original_shape, pad_image_to_block_size

DEFAULT_EPOCH_NUM = 60
DEFAULT_LAYER_NUM = 6
DEFAULT_SOLVER_FEATURE_DIM = 16
DEFAULT_PRIOR_FEATURE_CHANNELS = 96


def is_available() -> bool:
    return dependencies_available()


def resolve_rate_checkpoint_dir(checkpoint_dir: Path, cs_ratio: int) -> Path:
    """Resolve the required ``checkpoints/csXX`` distribution layout."""
    root = Path(checkpoint_dir)
    rate_name = f"cs{int(cs_ratio):02d}"
    if root.name.lower() == rate_name:
        return root
    return root / rate_name


def resolve_best_checkpoint_path(
    checkpoint_dir: Path,
    lambda_checkpoint_dir: Path,
    use_best: bool,
) -> Path:
    if not use_best:
        raise ValueError("resolve_best_checkpoint_path only supports use_best=True")

    direct_checkpoint_path = Path(checkpoint_dir) / BEST_CHECKPOINT_NAME
    if direct_checkpoint_path.exists() or Path(checkpoint_dir).name.upper().startswith("CS"):
        return direct_checkpoint_path

    lambda_checkpoint_path = Path(lambda_checkpoint_dir) / BEST_CHECKPOINT_NAME
    if lambda_checkpoint_path.exists():
        return lambda_checkpoint_path

    candidates = sorted(Path(checkpoint_dir).rglob(BEST_CHECKPOINT_NAME))
    if candidates:
        return candidates[0]

    return direct_checkpoint_path


def load_model(
    checkpoint_dir: Path,
    sampling_rate: float,
    device: str = "cpu",
    extra_args: dict[str, Any] | None = None,
) -> PreparedReferenceModel:
    require_dependencies()
    options = extra_args or {}
    cs_ratio = sampling_rate_to_cs_ratio(sampling_rate)
    checkpoint_dir = resolve_rate_checkpoint_dir(Path(checkpoint_dir), cs_ratio)
    measurement_count = measurement_count_from_ratio(cs_ratio)
    layer_num = int(options.get("layer_num", DEFAULT_LAYER_NUM))
    epoch_num = int(options.get("epoch_num", DEFAULT_EPOCH_NUM))
    learning_rate = float(options.get("learning_rate", 1e-4))
    group_num = int(options.get("group_num", 1))
    solver_feature_dim = int(options.get("solver_feature_dim", DEFAULT_SOLVER_FEATURE_DIM))
    prior_feature_channels = int(options.get("prior_feature_channels", DEFAULT_PRIOR_FEATURE_CHANNELS))
    transformer_depth = int(options.get("transformer_depth", 1))
    num_heads = int(options.get("num_heads", 4))
    window_size = int(options.get("window_size", 8))
    lambda_newton = float(options.get("lambda_newton", DEFAULT_LAMBDA_NEWTON))
    use_best = bool(options.get("use_best", False))
    coso_gaussian_fraction = float(options.get("coso_gaussian_fraction", 0.6))
    coso_filter_channels = int(options.get("coso_filter_channels", 8))
    coso_filter_residual_scale = float(options.get("coso_filter_residual_scale", 0.1))
    seed = int(options.get("seed", 2026))

    model_device = resolve_device(device)
    model = NDUNet(
        num_stages=effective_num_stages(layer_num, group_num),
        channels=1,
        solver_feature_dim=solver_feature_dim,
        prior_feature_channels=prior_feature_channels,
        transformer_depth=transformer_depth,
        num_heads=num_heads,
        window_size=window_size,
        measurement_count=measurement_count,
        operator_init_seed=seed,
        coso_gaussian_fraction=coso_gaussian_fraction,
        coso_filter_channels=coso_filter_channels,
        coso_filter_residual_scale=coso_filter_residual_scale,
    ).to(model_device)

    lambda_checkpoint_dir = target_checkpoint_dir(
        SimpleNamespace(
            checkpoint_dir=checkpoint_dir,
            layer_num=layer_num,
            group_num=group_num,
            prior_feature_channels=prior_feature_channels,
            transformer_depth=transformer_depth,
            num_heads=num_heads,
            window_size=window_size,
            block_size=BLOCK_SIZE,
            learning_rate=learning_rate,
            lambda_newton=lambda_newton,
            coso_gaussian_fraction=coso_gaussian_fraction,
            coso_filter_channels=coso_filter_channels,
            coso_filter_residual_scale=coso_filter_residual_scale,
            augmentation=bool(options.get("augmentation", True)),
        ),
        cs_ratio=cs_ratio,
    )
    if use_best:
        checkpoint_path = resolve_best_checkpoint_path(
            checkpoint_dir=checkpoint_dir,
            lambda_checkpoint_dir=lambda_checkpoint_dir,
            use_best=True,
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    else:
        checkpoint_search_dir = lambda_checkpoint_dir if lambda_checkpoint_dir.exists() else checkpoint_dir
        checkpoint_path = resolve_checkpoint_file(
            checkpoint_dir=checkpoint_search_dir,
            model_prefix=MODEL_PREFIX,
            layer_num=layer_num,
            group_num=group_num,
            cs_ratio=cs_ratio,
            learning_rate=learning_rate,
            epoch_num=epoch_num,
        )
    load_model_weights(model, checkpoint_path, device=model_device)
    model.eval()

    return PreparedReferenceModel(
        network=model,
        device=model_device,
        checkpoint_path=checkpoint_path,
        cs_ratio=cs_ratio,
        epoch_num=epoch_num,
        layer_num=layer_num,
        learning_rate=learning_rate,
        group_num=group_num,
    )


def reconstruct(
    image: np.ndarray,
    sampling_rate: float,
    block_size: int,
    checkpoint_dir: Path,
    device: str = "cpu",
    model: PreparedReferenceModel | None = None,
    image_path: Path | None = None,
    extra_args: dict[str, Any] | None = None,
) -> np.ndarray:
    del image_path
    require_dependencies()
    if block_size != BLOCK_SIZE:
        raise ValueError(f"{METHOD_NAME} expects block_size={BLOCK_SIZE}, got {block_size}")

    if model is None:
        model = load_model(
            checkpoint_dir=checkpoint_dir,
            sampling_rate=sampling_rate,
            device=device,
            extra_args=extra_args,
        )

    options = extra_args or {}
    del options
    prepared = model
    A, AT = prepared.network.A, prepared.network.AT
    image = np.asarray(image, dtype=np.float32)
    original_shape = image.shape[:2]
    padded, _ = pad_image_to_block_size(image, SAMPLING_BLOCK_SIZE)
    batch_x = torch.from_numpy(padded).float().unsqueeze(0).unsqueeze(0).to(prepared.device)

    with torch.no_grad():
        measurements = A(batch_x)
        x_init = AT(measurements)
        prediction, _ = prepared.network(measurements, A, AT, x_init=x_init)

    reconstruction = prediction.detach().cpu().numpy().astype(np.float32)[0, 0]
    return clip_image(crop_to_original_shape(reconstruction, original_shape))
