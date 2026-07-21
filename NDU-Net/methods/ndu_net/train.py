from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.dataset import find_image_paths, load_y_channel_image
from utils.metrics import FULL_Y_DATA_RANGE
from methods.ndu_net.coso import GlobalCollaborativeSamplingOperator

from methods.cs_reference import (
    BLOCK_SIZE,
    N_OUTPUT,
    SAMPLING_BLOCK_SIZE,
    load_model_weights,
    measurement_count_from_ratio,
    require_dependencies,
    resolve_checkpoint_file,
    resolve_device,
    sampling_rate_to_cs_ratio,
    torch,
)

ROOT = Path(__file__).resolve().parents[2]

if torch is not None:  # pragma: no branch
    import torch.nn as nn
    import torch.nn.functional as F
else:  # pragma: no cover
    nn = None
    F = None

MODEL_PREFIX = "CS_NDU_Net"
METHOD_NAME = "NDU-Net"
DEFAULT_EPOCH_NUM = 200
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 16
DEFAULT_LAMBDA_NEWTON = 0.01
DEFAULT_VALIDATION_SPLIT = 0.05
DEFAULT_MAX_TRAIN_SAMPLES = 0
DEFAULT_BLOCKS_PER_EPOCH = 8000
BEST_CHECKPOINT_NAME = "net_params_best.pkl"
BEST_METRICS_NAME = "net_params_best_metrics.json"


def soft_threshold(x, threshold):
    """Differentiable soft-thresholding with broadcastable thresholds."""
    return torch.sign(x) * F.relu(torch.abs(x) - threshold)


def _inverse_softplus(value: float) -> torch.Tensor:
    value_tensor = torch.tensor(float(value), dtype=torch.float32)
    if torch.any(value_tensor <= 0):
        raise ValueError(f"Expected a positive value for softplus initialization, got {value}")
    return torch.log(torch.expm1(value_tensor))


class LightweightNewtonSolverBlock(nn.Module):
    """Conv-ReLU-Conv-ReLU-Conv solver block for estimating Newton increments."""

    def __init__(
        self,
        in_dim=1,
        hidden_dim=16,
        kernel_size=3,
    ):
        super().__init__()
        padding = int(kernel_size) // 2
        self.body = nn.Sequential(
            nn.Conv2d(
                in_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                in_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            ),
        )

    def forward(self, residual):
        return self.body(residual)


class ApproxNewtonDCBlock(nn.Module):
    """One lightweight Newton-type data-consistency update stage."""

    def __init__(
        self,
        channels=1,
        feature_dim=16,
        eta_init=0.1,
    ):
        super().__init__()
        self.solver = LightweightNewtonSolverBlock(in_dim=channels, hidden_dim=feature_dim)
        self.log_eta = nn.Parameter(_inverse_softplus(float(eta_init)))

    def forward(self, x, y, A, AT):
        dc_residual = AT(A(x) - y)
        delta_x = self.solver(dc_residual)
        eta = F.softplus(self.log_eta)
        x_dc = x + eta * delta_x
        newton_residual = AT(A(delta_x)) + dc_residual
        aux = {
            "dc_residual": dc_residual,
            "delta_x": delta_x,
            "eta": eta,
            "newton_residual": newton_residual,
        }
        return x_dc, aux


class WindowTransformerBlock(nn.Module):
    """Window self-attention block for feature-level image restoration priors."""

    def __init__(
        self,
        dim=96,
        num_heads=4,
        window_size=8,
        mlp_ratio=2,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")

        hidden_dim = int(self.dim) * int(mlp_ratio)
        self.norm1 = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, self.dim, bias=True),
        )

    def _window_partition(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        x = x.view(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(-1, ws * ws, c)

    def _window_reverse(self, windows, batch_size, channels, height, width):
        ws = self.window_size
        x = windows.view(batch_size, height // ws, width // ws, ws, ws, channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(batch_size, channels, height, width)

    def forward(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, hp, wp = x.shape
        windows = self._window_partition(x)
        residual = windows
        windows = self.norm1(windows)

        qkv = self.qkv(windows).view(windows.shape[0], windows.shape[1], 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        attention = torch.matmul(q, k.transpose(-2, -1)) * ((c // self.num_heads) ** -0.5)
        attention = attention.softmax(dim=-1)
        attended = torch.matmul(attention, v).permute(0, 2, 1, 3).contiguous().view(windows.shape[0], windows.shape[1], c)
        windows = residual + self.proj(attended)
        windows = windows + self.mlp(self.norm2(windows))

        x = self._window_reverse(windows, b, c, hp, wp)
        return x[:, :, :h, :w]


class NewtonGuidedMSCTPriorBlock(nn.Module):
    """Newton-guided MSCT sparse prior with direct additive guidance."""

    def __init__(
        self,
        channels=1,
        feature_channels=96,
        transformer_depth=1,
        num_heads=4,
        window_size=8,
        tau_init=0.01,
    ):
        super().__init__()
        self.guidance_encoder = nn.Sequential(
            nn.Conv2d(3 * channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.embedding = nn.Sequential(
            nn.Conv2d(channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.middle_embed = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.middle_transformers = nn.ModuleList(
            [
                WindowTransformerBlock(
                    dim=feature_channels,
                    num_heads=num_heads,
                    window_size=window_size,
                )
                for _ in range(int(transformer_depth))
            ]
        )
        self.large_embed = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.large_transformers = nn.ModuleList(
            [
                WindowTransformerBlock(
                    dim=feature_channels,
                    num_heads=num_heads,
                    window_size=window_size,
                )
                for _ in range(int(transformer_depth))
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(3 * feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.synthesis_transform = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, channels, kernel_size=3, padding=1, bias=False),
        )
        self.log_tau = nn.Parameter(_inverse_softplus(float(tau_init)).view(1, 1, 1, 1).repeat(1, feature_channels, 1, 1))

    def forward(self, x_dc, dc_residual, delta_x, newton_residual):
        guidance_input = torch.cat([delta_x, dc_residual, newton_residual], dim=1)
        guidance_feature = self.guidance_encoder(guidance_input)

        embedded = self.embedding(x_dc)
        local_feature = self.local_branch(embedded)

        middle_feature = self.middle_embed(F.avg_pool2d(embedded, kernel_size=2, stride=2))
        for block in self.middle_transformers:
            middle_feature = block(middle_feature)
        middle_feature = F.interpolate(middle_feature, size=x_dc.shape[-2:], mode="bilinear", align_corners=False)

        large_feature = self.large_embed(F.avg_pool2d(embedded, kernel_size=4, stride=4))
        for block in self.large_transformers:
            large_feature = block(large_feature)
        large_feature = F.interpolate(large_feature, size=x_dc.shape[-2:], mode="bilinear", align_corners=False)

        feature = self.fusion(torch.cat([local_feature, middle_feature, large_feature], dim=1))
        guided_feature = feature + guidance_feature
        tau = F.softplus(self.log_tau)
        feature_shrink = soft_threshold(guided_feature, tau)
        residual_update = self.synthesis_transform(feature_shrink)
        x_next = x_dc + residual_update

        aux = {
            "guidance_feature": guidance_feature,
            "local_feature": local_feature,
            "middle_feature": middle_feature,
            "large_feature": large_feature,
            "feature": feature,
            "guided_feature": guided_feature,
            "feature_shrink": feature_shrink,
            "residual_update": residual_update,
            "tau": tau,
            "dc_residual_abs_mean": dc_residual.abs().mean(),
            "delta_abs_mean": delta_x.abs().mean(),
            "newton_residual_abs_mean": newton_residual.abs().mean(),
        }
        return x_next, aux


class NDUNet(nn.Module):
    """Newton-driven unfolding network with message-guided multi-scale priors."""

    def __init__(
        self,
        num_stages=9,
        channels=1,
        solver_feature_dim=16,
        prior_feature_channels=96,
        transformer_depth=1,
        num_heads=4,
        window_size=8,
        eta_init=0.1,
        tau_init=0.01,
        measurement_count: int | None = None,
        operator_init_seed: int = 2026,
        sampling_block_size: int = SAMPLING_BLOCK_SIZE,
        coso_gaussian_fraction: float = 0.6,
        coso_filter_channels: int = 8,
        coso_filter_residual_scale: float = 0.1,
    ):
        super().__init__()
        self.num_stages = int(num_stages)
        if measurement_count is None:
            raise ValueError("measurement_count is required for the COSO sampling operator")
        self.cs_operator = GlobalCollaborativeSamplingOperator(
            measurement_count=int(measurement_count),
            sampling_block_size=int(sampling_block_size),
            gaussian_fraction=float(coso_gaussian_fraction),
            filter_channels=int(coso_filter_channels),
            filter_residual_scale=float(coso_filter_residual_scale),
            seed=int(operator_init_seed),
        )
        self.dc_blocks = nn.ModuleList(
            [
                ApproxNewtonDCBlock(
                    channels=channels,
                    feature_dim=solver_feature_dim,
                    eta_init=eta_init,
                )
                for _ in range(self.num_stages)
            ]
        )
        self.prior_blocks = nn.ModuleList(
            [
                NewtonGuidedMSCTPriorBlock(
                    channels=channels,
                    feature_channels=prior_feature_channels,
                    transformer_depth=transformer_depth,
                    num_heads=num_heads,
                    window_size=window_size,
                    tau_init=tau_init,
                )
                for _ in range(self.num_stages)
            ]
        )

    def forward(self, y, A, AT, x_init=None):
        x = x_init if x_init is not None else AT(y)
        aux_list = []

        for dc_block, prior_block in zip(self.dc_blocks, self.prior_blocks):
            x_dc, dc_aux = dc_block(x, y, A, AT)
            x, prior_aux = prior_block(
                x_dc,
                dc_aux["dc_residual"],
                dc_aux["delta_x"],
                dc_aux["newton_residual"],
            )
            aux_list.append(
                {
                    "dc": dc_aux,
                    "prior": prior_aux,
                    "x_stage": x,
                }
            )

        return x, aux_list

    def A(self, x):
        return self.cs_operator.A(x)

    def AT(self, y):
        return self.cs_operator.AT(y)


def effective_num_stages(layer_num: int, group_num: int) -> int:
    return max(1, int(layer_num)) * max(1, int(group_num))


def build_model_from_args(args: argparse.Namespace) -> NDUNet:
    cs_ratio = sampling_rate_to_cs_ratio(args.sampling_rate)
    measurement_count = measurement_count_from_ratio(cs_ratio)
    return NDUNet(
        num_stages=effective_num_stages(args.layer_num, args.group_num),
        channels=1,
        solver_feature_dim=int(args.solver_feature_dim),
        prior_feature_channels=int(args.prior_feature_channels),
        transformer_depth=int(getattr(args, "transformer_depth", 1)),
        num_heads=int(getattr(args, "num_heads", 4)),
        window_size=int(getattr(args, "window_size", 8)),
        eta_init=float(args.eta_init),
        tau_init=float(args.tau_init),
        measurement_count=measurement_count,
        operator_init_seed=int(getattr(args, "seed", 2026)),
        coso_gaussian_fraction=float(getattr(args, "coso_gaussian_fraction", 0.6)),
        coso_filter_channels=int(getattr(args, "coso_filter_channels", 8)),
        coso_filter_residual_scale=float(getattr(args, "coso_filter_residual_scale", 0.1)),
    )


class TrainingDataset(Dataset):
    def __init__(self, labels: np.ndarray):
        self.labels = labels

    def __getitem__(self, index):
        block = torch.from_numpy(self.labels[index, :]).float()
        return block.view(1, BLOCK_SIZE, BLOCK_SIZE)

    def __len__(self):
        return int(self.labels.shape[0])


def augment_y_channel_block(block: np.ndarray, mode: int) -> np.ndarray:
    """Apply one of the eight dihedral rotations/reflections to a Y patch."""
    mode = int(mode)
    if mode < 0 or mode > 7:
        raise ValueError(f"augmentation mode must be in [0, 7], got {mode}")
    transformed = np.fliplr(block) if mode >= 4 else block
    transformed = np.rot90(transformed, k=mode % 4)
    return np.ascontiguousarray(transformed, dtype=np.float32)


class RandomYChannelBlockDataset(Dataset):
    """Random Y-channel patches from an image directory."""

    def __init__(
        self,
        image_paths: list[Path],
        block_size: int = BLOCK_SIZE,
        blocks_per_epoch: int = DEFAULT_BLOCKS_PER_EPOCH,
        seed: int = 2026,
        deterministic: bool = False,
        augment: bool = False,
    ):
        if not image_paths:
            raise ValueError("RandomYChannelBlockDataset requires at least one image.")
        self.image_paths = [Path(path) for path in image_paths]
        self.block_size = int(block_size)
        self.blocks_per_epoch = int(blocks_per_epoch)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.augment = bool(augment)
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.blocks_per_epoch <= 0:
            raise ValueError(f"blocks_per_epoch must be positive, got {self.blocks_per_epoch}")

    def __len__(self):
        return self.blocks_per_epoch

    def __getitem__(self, index):
        if self.deterministic:
            rng = np.random.default_rng(self.seed + int(index))
        else:
            rng_seed = int(torch.empty((), dtype=torch.int64).random_().item())
            rng = np.random.default_rng(rng_seed)
        image_path = self.image_paths[int(rng.integers(0, len(self.image_paths)))]
        image = load_y_channel_image(image_path)
        if image.shape[0] < self.block_size or image.shape[1] < self.block_size:
            pad_h = max(0, self.block_size - image.shape[0])
            pad_w = max(0, self.block_size - image.shape[1])
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="edge")
        top = int(rng.integers(0, image.shape[0] - self.block_size + 1))
        left = int(rng.integers(0, image.shape[1] - self.block_size + 1))
        block = np.asarray(image[top : top + self.block_size, left : left + self.block_size], dtype=np.float32)
        if self.augment:
            block = augment_y_channel_block(block, int(rng.integers(0, 8)))
        return torch.from_numpy(block).unsqueeze(0)


RandomGrayscaleBlockDataset = RandomYChannelBlockDataset


def normalize_labels_array(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError(f"Training labels must be a 2D array, got shape {labels.shape}")
    if labels.shape[1] == N_OUTPUT:
        return np.asarray(labels, dtype=np.float32)
    if labels.shape[0] == N_OUTPUT:
        return np.asarray(labels.transpose(), dtype=np.float32)
    raise ValueError(f"Training labels must contain {N_OUTPUT}-dimensional blocks, got {labels.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train NDU-Net for patch-CS reconstruction.")
    parser.add_argument("--training-image-dir", type=Path, default=ROOT / "data" / "WED")
    parser.add_argument("--validation-image-dir", type=Path, default=None)
    parser.add_argument("--training-data-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--sampling-rate", type=float, default=0.25)
    parser.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    parser.add_argument("--blocks-per-epoch", type=int, default=DEFAULT_BLOCKS_PER_EPOCH)
    parser.add_argument("--validation-blocks", type=int, default=1024)
    parser.add_argument("--coso-gaussian-fraction", type=float, default=0.6)
    parser.add_argument("--coso-filter-channels", type=int, default=8)
    parser.add_argument(
        "--coso-filter-residual-scale",
        type=float,
        default=0.1,
        help="Initial value of the learnable positive COSO residual scale.",
    )
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--end-epoch", type=int, default=DEFAULT_EPOCH_NUM)
    parser.add_argument("--layer-num", type=int, default=6)
    parser.add_argument("--solver-feature-dim", type=int, default=16)
    parser.add_argument("--prior-feature-channels", type=int, default=96)
    parser.add_argument("--transformer-depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--eta-init", type=float, default=1.0)
    parser.add_argument("--tau-init", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--group-num", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-split", type=float, default=DEFAULT_VALIDATION_SPLIT)
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--verbose-best-checkpoint", action="store_true")
    parser.add_argument("--lambda-newton", type=float, default=DEFAULT_LAMBDA_NEWTON)
    parser.add_argument("--lambda-newton-schedule", choices=("piecewise", "constant"), default="piecewise")
    parser.add_argument("--lambda-newton-stage1-end", type=int, default=100)
    parser.add_argument("--lambda-newton-stage2-end", type=int, default=150)
    parser.add_argument("--lambda-newton-stage2", type=float, default=0.005)
    parser.add_argument("--lambda-newton-stage3", type=float, default=0.001)
    parser.add_argument("--lambda-dc", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.set_defaults(augmentation=True)
    return parser


def lambda_newton_for_epoch(args: argparse.Namespace, epoch_idx: int) -> float:
    """Return the Newton residual weight used for the current epoch."""
    base_lambda = float(args.lambda_newton)
    if getattr(args, "lambda_newton_schedule", "piecewise") == "constant":
        return base_lambda

    if int(epoch_idx) <= int(getattr(args, "lambda_newton_stage1_end", 100)):
        return base_lambda
    if int(epoch_idx) <= int(getattr(args, "lambda_newton_stage2_end", 150)):
        return float(getattr(args, "lambda_newton_stage2", 0.005))
    return float(getattr(args, "lambda_newton_stage3", 0.001))


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def compute_training_loss(
    *,
    x_pred: torch.Tensor,
    x_gt: torch.Tensor,
    y: torch.Tensor,
    A,
    AT,
    aux_list: list[dict],
    lambda_newton: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_recon = F.l1_loss(x_pred, x_gt)
    del y, A, AT
    newton_residuals = [item["dc"]["newton_residual"].abs().mean() for item in aux_list]
    loss_newton = torch.stack(newton_residuals).mean()
    loss = loss_recon + float(lambda_newton) * loss_newton
    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_newton": float(loss_newton.detach().cpu()),
    }
    return loss, metrics


def split_train_validation_labels(
    labels: np.ndarray,
    *,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    seed: int = 2026,
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Split block-level training labels into train and validation subsets."""
    if labels.ndim != 2:
        raise ValueError(f"labels must be a 2D array, got shape {labels.shape}")
    indices = np.arange(labels.shape[0])
    np.random.default_rng(int(seed)).shuffle(indices)
    if int(max_train_samples) > 0:
        indices = indices[: min(int(max_train_samples), len(indices))]
    if len(indices) < 2 and float(validation_split) > 0.0:
        raise ValueError("At least two training blocks are required when validation_split > 0.")

    if float(validation_split) <= 0.0:
        validation_count = 0
    else:
        validation_count = round(len(indices) * float(validation_split))
        validation_count = max(1, min(validation_count, len(indices) - 1))

    val_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    split_info = {
        "total_blocks": int(len(indices)),
        "training_blocks": int(len(train_indices)),
        "validation_blocks": int(len(val_indices)),
    }
    return labels[train_indices], labels[val_indices], split_info


def split_train_validation_image_paths(
    image_paths: list[Path],
    *,
    validation_split: float = DEFAULT_VALIDATION_SPLIT,
    seed: int = 2026,
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES,
) -> tuple[list[Path], list[Path], dict[str, int]]:
    """Split image paths into training and validation subsets."""
    paths = [Path(path) for path in image_paths]
    indices = np.arange(len(paths))
    np.random.default_rng(int(seed)).shuffle(indices)
    if int(max_train_samples) > 0:
        indices = indices[: min(int(max_train_samples), len(indices))]
    if len(indices) < 2 and float(validation_split) > 0.0:
        raise ValueError("At least two images are required when validation_split > 0.")

    if float(validation_split) <= 0.0:
        validation_count = 0
    else:
        validation_count = round(len(indices) * float(validation_split))
        validation_count = max(1, min(validation_count, len(indices) - 1))

    val_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    train_paths = [paths[index] for index in train_indices]
    val_paths = [paths[index] for index in val_indices]
    split_info = {
        "total_images": int(len(indices)),
        "training_images": int(len(train_paths)),
        "validation_images": int(len(val_paths)),
    }
    return train_paths, val_paths, split_info


def build_train_validation_loaders(
    train_image_paths: list[Path],
    val_image_paths: list[Path],
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader | None, dict[str, int]]:
    train_paths = list(train_image_paths)
    val_paths = list(val_image_paths)
    split_info = {
        "training_images": int(len(train_paths)),
        "validation_images": int(len(val_paths)),
    }
    train_loader = DataLoader(
        dataset=RandomYChannelBlockDataset(
            train_paths,
            block_size=BLOCK_SIZE,
            blocks_per_epoch=int(args.blocks_per_epoch),
            seed=int(args.seed),
            deterministic=False,
            augment=bool(getattr(args, "augmentation", True)),
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )
    val_loader = None
    if len(val_paths) > 0:
        val_loader = DataLoader(
            dataset=RandomYChannelBlockDataset(
                val_paths,
                block_size=BLOCK_SIZE,
                blocks_per_epoch=int(args.validation_blocks),
                seed=int(args.seed) + 100000,
                deterministic=True,
                augment=False,
            ),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )
    return train_loader, val_loader, split_info


def limit_image_paths(
    image_paths: list[Path],
    max_train_samples: int,
    seed: int,
) -> list[Path]:
    paths = [Path(path) for path in image_paths]
    if int(max_train_samples) <= 0:
        return paths
    indices = np.arange(len(paths))
    np.random.default_rng(int(seed)).shuffle(indices)
    indices = indices[: min(int(max_train_samples), len(indices))]
    return [paths[index] for index in indices]


def average_metric_totals(total_metrics: dict[str, float], batch_count: int) -> dict[str, float]:
    return {key: value / max(batch_count, 1) for key, value in total_metrics.items()}


def batch_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    data_range: float = FULL_Y_DATA_RANGE,
) -> torch.Tensor:
    mse = (prediction - target).flatten(1).pow(2).mean(dim=1)
    return 20.0 * torch.log10(prediction.new_tensor(float(data_range))) - 10.0 * torch.log10(mse.clamp_min(1e-12))


def evaluate_validation_loader(
    model: NDUNet,
    loader: DataLoader | None,
    A,
    AT,
    device,
) -> dict[str, float]:
    if loader is None:
        return {}
    model.eval()
    total_psnr = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch_x in loader:
            batch_x = batch_x.to(device)
            y = A(batch_x)
            x_init = AT(y)
            x_pred, _ = model(y, A, AT, x_init=x_init)
            psnr = batch_psnr(x_pred, batch_x)
            total_psnr += float(psnr.sum().detach().cpu())
            sample_count += int(batch_x.shape[0])
    if sample_count <= 0:
        return {}
    return {"psnr": total_psnr / sample_count}


def maybe_save_best_checkpoint(
    *,
    model: nn.Module,
    target_dir: Path,
    epoch_idx: int,
    val_metrics: dict[str, float],
    best_val_psnr: float,
    verbose: bool = False,
) -> float:
    """Save the model state when validation PSNR improves."""
    if not val_metrics:
        return float(best_val_psnr)
    current_val_psnr = float(val_metrics.get("psnr", float("-inf")))
    if current_val_psnr > float(best_val_psnr):
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), target_dir / BEST_CHECKPOINT_NAME)
        metrics_payload = {
            "epoch": int(epoch_idx),
            "val_psnr": current_val_psnr,
            "val_metrics": {key: float(value) for key, value in val_metrics.items()},
        }
        (target_dir / BEST_METRICS_NAME).write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        if bool(verbose):
            print(
                f"[{int(epoch_idx):03d}] Saved best checkpoint with val_psnr={current_val_psnr:.4f}",
                flush=True,
            )
        return current_val_psnr
    return float(best_val_psnr)


def load_best_val_psnr(target_dir: Path) -> float:
    metrics_path = Path(target_dir) / BEST_METRICS_NAME
    if not metrics_path.exists():
        return float("-inf")
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(payload.get("val_psnr", float("-inf")))
    except (OSError, ValueError, TypeError):
        return float("-inf")


def target_checkpoint_dir(args: argparse.Namespace, cs_ratio: int) -> Path:
    root = Path(args.checkpoint_dir)
    rate_name = f"cs{int(cs_ratio):02d}"
    return root if root.name.lower() == rate_name else root / rate_name


def train_model(args: argparse.Namespace) -> None:
    require_dependencies()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    if int(args.block_size) != BLOCK_SIZE:
        raise ValueError(f"{METHOD_NAME} is configured for block_size={BLOCK_SIZE}, got {args.block_size}")
    cs_ratio = sampling_rate_to_cs_ratio(args.sampling_rate)

    train_image_paths = find_image_paths(args.training_image_dir)
    if not train_image_paths:
        raise FileNotFoundError(f"No training images found under {args.training_image_dir}")
    if args.validation_image_dir is None:
        train_image_paths, val_image_paths, split_info = split_train_validation_image_paths(
            train_image_paths,
            validation_split=float(args.validation_split),
            seed=int(args.seed),
            max_train_samples=int(args.max_train_samples),
        )
    else:
        train_image_paths = limit_image_paths(train_image_paths, int(args.max_train_samples), int(args.seed))
        val_image_paths = find_image_paths(args.validation_image_dir)
        if not val_image_paths:
            raise FileNotFoundError(f"No validation images found under {args.validation_image_dir}")
        split_info = {
            "total_images": int(len(train_image_paths) + len(val_image_paths)),
            "training_images": int(len(train_image_paths)),
            "validation_images": int(len(val_image_paths)),
        }

    train_loader, val_loader, loader_info = build_train_validation_loaders(train_image_paths, val_image_paths, args)
    split_info.update(loader_info)
    print(
        f"[{METHOD_NAME}] image-split patch-CS protocol: "
        f"total_images={split_info.get('total_images', 0)}, "
        f"train_images={split_info['training_images']}, "
        f"val_images={split_info['validation_images']}, "
        f"validation_split={float(args.validation_split):g}, "
        f"block_size={BLOCK_SIZE}, "
        f"blocks_per_epoch={int(args.blocks_per_epoch)}, "
        f"validation_blocks={int(args.validation_blocks)}, "
        f"sampling_rate={float(args.sampling_rate):g}, "
        f"augmentation={bool(getattr(args, 'augmentation', True))}, "
        "sampling_operator=global_coso_merged_adjoint_learnable_alpha",
        flush=True,
    )

    model = build_model_from_args(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    target_dir = target_checkpoint_dir(args, cs_ratio)
    target_dir.mkdir(parents=True, exist_ok=True)

    if int(args.start_epoch) > 0:
        checkpoint_path = resolve_checkpoint_file(
            checkpoint_dir=target_dir,
            model_prefix=MODEL_PREFIX,
            layer_num=int(args.layer_num),
            group_num=int(args.group_num),
            cs_ratio=cs_ratio,
            learning_rate=float(args.learning_rate),
            epoch_num=int(args.start_epoch),
        )
        load_model_weights(model, checkpoint_path, device=device)

    A, AT = model.A, model.AT
    best_val_psnr = load_best_val_psnr(target_dir)

    for epoch_idx in range(int(args.start_epoch) + 1, int(args.end_epoch) + 1):
        current_lambda_newton = lambda_newton_for_epoch(args, epoch_idx)
        model.train()
        total_metrics: dict[str, float] = {}
        batch_count = 0

        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            y = A(batch_x)
            x_init = AT(y)
            x_pred, aux_list = model(y, A, AT, x_init=x_init)
            loss, metrics = compute_training_loss(
                x_pred=x_pred,
                x_gt=batch_x,
                y=y,
                A=A,
                AT=AT,
                aux_list=aux_list,
                lambda_newton=float(current_lambda_newton),
            )

            optimizer.zero_grad()
            loss.backward()
            if float(args.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
            optimizer.step()

            for key, value in metrics.items():
                total_metrics[key] = total_metrics.get(key, 0.0) + value
            batch_count += 1

        averaged = average_metric_totals(total_metrics, batch_count)
        val_metrics = evaluate_validation_loader(
            model,
            val_loader,
            A,
            AT,
            device,
        )
        best_val_psnr = maybe_save_best_checkpoint(
            model=model,
            target_dir=target_dir,
            epoch_idx=epoch_idx,
            val_metrics=val_metrics,
            best_val_psnr=best_val_psnr,
            verbose=bool(getattr(args, "verbose_best_checkpoint", False)),
        )
        val_text = ""
        if val_metrics:
            val_text = f" val_psnr: {val_metrics.get('psnr', 0.0):.4f}"
        print(
            f"[{epoch_idx:03d}/{int(args.end_epoch):03d}] "
            f"{METHOD_NAME} loss: {averaged.get('loss', 0.0):.6f} "
            f"recon: {averaged.get('loss_recon', 0.0):.6f} "
            f"newton: {averaged.get('loss_newton', 0.0):.6f}"
            f" lambda_newton: {current_lambda_newton:g}"
            f"{val_text}",
            flush=True,
        )

        if int(args.checkpoint_interval) > 0 and epoch_idx % int(args.checkpoint_interval) == 0:
            torch.save(model.state_dict(), target_dir / f"net_params_{epoch_idx}.pkl")


def test_forward():
    B, C, H, W = 2, 1, BLOCK_SIZE, BLOCK_SIZE
    x_gt = torch.randn(B, C, H, W)

    def A(x):
        return x

    def AT(y):
        return y

    y = A(x_gt)

    model = NDUNet(
        num_stages=3,
        channels=1,
        solver_feature_dim=16,
        prior_feature_channels=32,
        transformer_depth=1,
        num_heads=4,
        window_size=8,
        eta_init=1.0,
        tau_init=0.01,
        measurement_count=measurement_count_from_ratio(25),
    )

    x_pred, aux_list = model(y, A, AT)

    print("x_pred shape:", x_pred.shape)
    print("num stages:", len(aux_list))
    print("dc_residual stage 0:", aux_list[0]["dc"]["dc_residual"].abs().mean())
    print("eta stage 0:", aux_list[0]["dc"]["eta"])
    print("newton_residual stage 0:", aux_list[0]["dc"]["newton_residual"].abs().mean())
    print("guidance stage 0:", aux_list[0]["prior"]["guidance_feature"].abs().mean())


def main() -> None:
    args = build_parser().parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
