from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zigzag_flat_indices(height: int, width: int, device: torch.device) -> torch.Tensor:
    indices: list[int] = []
    for diagonal in range(int(height) + int(width) - 1):
        row_min = max(0, diagonal - int(width) + 1)
        row_max = min(int(height) - 1, diagonal)
        rows = range(row_max, row_min - 1, -1) if diagonal % 2 == 0 else range(row_min, row_max + 1)
        for row in rows:
            column = diagonal - row
            indices.append(row * int(width) + column)
    return torch.tensor(indices, dtype=torch.long, device=device)


def _orthonormal_dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(int(size), device=device, dtype=torch.float64).unsqueeze(0)
    frequencies = torch.arange(int(size), device=device, dtype=torch.float64).unsqueeze(1)
    matrix = torch.cos(math.pi * (positions + 0.5) * frequencies / float(size))
    matrix[0] *= math.sqrt(1.0 / float(size))
    matrix[1:] *= math.sqrt(2.0 / float(size))
    return matrix.to(dtype=dtype)


class IdentityInitializedConditionalFilter(nn.Module):
    """Bias-free conditional filter with an identity path and learnable scale."""

    def __init__(
        self,
        channels: int,
        ratio_vector: tuple[float, float],
        residual_scale: float = 0.1,
    ):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if residual_scale <= 0:
            raise ValueError(f"residual_scale must be positive, got {residual_scale}")

        self.head = nn.Conv2d(1, channels, 3, padding=1, bias=False)
        self.body = nn.ModuleList(
            [nn.Conv2d(channels, channels, 3, padding=1, bias=False) for _ in range(5)]
        )
        self.conditioners = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2, channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(channels, channels),
                )
                for _ in range(5)
            ]
        )
        self.tail = nn.Conv2d(channels, 2, 3, padding=1, bias=False)
        self.register_buffer("ratio_vector", torch.tensor(ratio_vector, dtype=torch.float32))
        initial_log_scale = math.log(math.expm1(float(residual_scale)))
        self.log_residual_scale = nn.Parameter(torch.tensor(initial_log_scale, dtype=torch.float32))

        for conditioner in self.conditioners:
            nn.init.zeros_(conditioner[-1].weight)
            nn.init.ones_(conditioner[-1].bias)

    def _scales(self, reference: torch.Tensor) -> list[torch.Tensor]:
        ratio = self.ratio_vector.to(device=reference.device, dtype=reference.dtype).unsqueeze(0)
        return [conditioner(ratio).view(1, -1, 1, 1) for conditioner in self.conditioners]

    def _residual(self, image: torch.Tensor) -> torch.Tensor:
        features = self.head(image)
        for convolution, scale in zip(self.body, self._scales(image)):
            features = scale * convolution(features)
        return self.tail(features)

    def residual_scale(self, reference: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.log_residual_scale).to(device=reference.device, dtype=reference.dtype)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        identity = torch.cat([image, image], dim=1)
        scale = self.residual_scale(image)
        return identity + scale * self._residual(image)

    def adjoint(self, filtered: torch.Tensor) -> torch.Tensor:
        identity_adjoint = filtered[:, 0:1] + filtered[:, 1:2]
        features = F.conv_transpose2d(filtered, self.tail.weight, padding=1)
        scales = self._scales(filtered)
        for convolution, scale in zip(reversed(self.body), reversed(scales)):
            features = F.conv_transpose2d(scale * features, convolution.weight, padding=1)
        residual_adjoint = F.conv_transpose2d(features, self.head.weight, padding=1)
        scale = self.residual_scale(filtered)
        return identity_adjoint + scale * residual_adjoint


class GlobalCollaborativeSamplingOperator(nn.Module):
    """Single-rate full-image COSO with merged exact adjoint.

    The DCT and pixel permutation operate on the complete padded image. The
    Gaussian transform remains a fast 32x32 block convolution after the global
    permutation. Both branch adjoints are added inside ``AT`` so NDU-Net keeps
    one data-consistency path.
    """

    def __init__(
        self,
        measurement_count: int,
        sampling_block_size: int = 32,
        gaussian_fraction: float = 0.6,
        filter_channels: int = 8,
        filter_residual_scale: float = 0.1,
        seed: int = 2026,
    ):
        super().__init__()
        self.measurement_count = int(measurement_count)
        self.sampling_block_size = int(sampling_block_size)
        self.gaussian_fraction = float(gaussian_fraction)
        self.filter_channels = int(filter_channels)
        self.seed = int(seed)
        block_elements = self.sampling_block_size**2

        if not 0 < self.measurement_count <= block_elements:
            raise ValueError(
                f"measurement_count must be in [1, {block_elements}], got {self.measurement_count}"
            )
        if not 0.0 <= self.gaussian_fraction <= 1.0:
            raise ValueError(f"gaussian_fraction must be in [0, 1], got {self.gaussian_fraction}")

        if self.measurement_count <= block_elements // 2:
            self.gaussian_count = int(round(self.gaussian_fraction * self.measurement_count))
        else:
            self.gaussian_count = self.measurement_count
        self.dct_count = self.measurement_count - self.gaussian_count

        self.filter = IdentityInitializedConditionalFilter(
            channels=self.filter_channels,
            ratio_vector=(
                self.gaussian_count / float(block_elements),
                self.dct_count / float(block_elements),
            ),
            residual_scale=float(filter_residual_scale),
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.measurement_count)
        if self.gaussian_count:
            raw = torch.randn(block_elements, block_elements, generator=generator)
            orthogonal, _ = torch.linalg.qr(raw, mode="reduced")
            row_order = torch.randperm(block_elements, generator=generator)
            gaussian = orthogonal.t()[row_order[: self.gaussian_count]].contiguous()
            gaussian_weight = gaussian.reshape(
                self.gaussian_count, 1, self.sampling_block_size, self.sampling_block_size
            )
        else:
            gaussian_weight = torch.empty(
                (0, 1, self.sampling_block_size, self.sampling_block_size), dtype=torch.float32
            )
        self.register_buffer("gaussian_weight", gaussian_weight)

        self._active_shape: tuple[int, int] | None = None
        self._dct_cache: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}
        self._zigzag_cache: dict[tuple[int, int, str], torch.Tensor] = {}
        self._permutation_cache: dict[tuple[int, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _validate_shape(self, height: int, width: int) -> None:
        if height % self.sampling_block_size or width % self.sampling_block_size:
            raise ValueError(
                f"COSO input shape {(height, width)} must be divisible by {self.sampling_block_size}"
            )

    def _measurement_sizes(self, height: int, width: int) -> tuple[int, int]:
        blocks = (int(height) // self.sampling_block_size) * (int(width) // self.sampling_block_size)
        return self.gaussian_count * blocks, self.dct_count * blocks

    def total_measurements(self, height: int, width: int) -> int:
        gaussian, dct = self._measurement_sizes(height, width)
        return gaussian + dct

    def _dct_matrix(self, size: int, reference: torch.Tensor) -> torch.Tensor:
        key = (int(size), str(reference.device), reference.dtype)
        if key not in self._dct_cache:
            self._dct_cache[key] = _orthonormal_dct_matrix(
                int(size), reference.device, reference.dtype
            )
        return self._dct_cache[key]

    def _zigzag(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (int(height), int(width), str(device))
        if key not in self._zigzag_cache:
            self._zigzag_cache[key] = _zigzag_flat_indices(height, width, device)
        return self._zigzag_cache[key]

    def _permutations(
        self, height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(height), int(width), str(device))
        if key not in self._permutation_cache:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.seed + int(height) * 1_000_003 + int(width))
            permutation = torch.randperm(int(height) * int(width), generator=generator).to(device)
            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(permutation.numel(), device=device)
            self._permutation_cache[key] = (permutation, inverse)
        return self._permutation_cache[key]

    def _dct2(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        matrix_h = self._dct_matrix(height, image)
        matrix_w = self._dct_matrix(width, image)
        return torch.matmul(torch.matmul(matrix_h, image), matrix_w.t())

    def _idct2(self, coefficients: torch.Tensor) -> torch.Tensor:
        height, width = coefficients.shape[-2:]
        matrix_h = self._dct_matrix(height, coefficients)
        matrix_w = self._dct_matrix(width, coefficients)
        return torch.matmul(torch.matmul(matrix_h.t(), coefficients), matrix_w)

    def A(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"COSO expects [B, 1, H, W], got {tuple(image.shape)}")
        batch, _, height, width = image.shape
        self._validate_shape(height, width)
        self._active_shape = (int(height), int(width))
        filtered = self.filter(image)
        gaussian_size, dct_size = self._measurement_sizes(height, width)
        measurements: list[torch.Tensor] = []

        if gaussian_size:
            permutation, _ = self._permutations(height, width, image.device)
            scrambled = filtered[:, 1:2].flatten(2)[:, :, permutation]
            scrambled = scrambled.reshape(batch, 1, height, width)
            gaussian = F.conv2d(
                scrambled,
                self.gaussian_weight.to(dtype=image.dtype),
                stride=self.sampling_block_size,
            )
            measurements.append(gaussian.flatten(1))

        if dct_size:
            coefficients = self._dct2(filtered[:, 0:1]).flatten(1)
            selected = self._zigzag(height, width, image.device)[:dct_size]
            measurements.append(coefficients[:, selected])

        return torch.cat(measurements, dim=1)

    def AT(self, measurements: torch.Tensor) -> torch.Tensor:
        if self._active_shape is None:
            raise RuntimeError("A(image) must be called before AT(measurements) to configure image shape")
        height, width = self._active_shape
        expected = self.total_measurements(height, width)
        if measurements.ndim != 2 or measurements.shape[1] != expected:
            raise ValueError(f"COSO expects [B, {expected}], got {tuple(measurements.shape)}")

        batch = measurements.shape[0]
        gaussian_size, dct_size = self._measurement_sizes(height, width)
        filtered_adjoint = measurements.new_zeros(batch, 2, height, width)
        offset = 0

        if gaussian_size:
            gaussian = measurements[:, :gaussian_size].reshape(
                batch,
                self.gaussian_count,
                height // self.sampling_block_size,
                width // self.sampling_block_size,
            )
            offset = gaussian_size
            scrambled = F.conv_transpose2d(
                gaussian,
                self.gaussian_weight.to(dtype=measurements.dtype),
                stride=self.sampling_block_size,
            )
            _, inverse = self._permutations(height, width, measurements.device)
            restored = scrambled.flatten(2)[:, :, inverse]
            filtered_adjoint[:, 1:2] = restored.reshape(batch, 1, height, width)

        if dct_size:
            coefficients = measurements.new_zeros(batch, height * width)
            selected = self._zigzag(height, width, measurements.device)[:dct_size]
            coefficients[:, selected] = measurements[:, offset:]
            filtered_adjoint[:, 0:1] = self._idct2(
                coefficients.reshape(batch, 1, height, width)
            )

        return self.filter.adjoint(filtered_adjoint)
