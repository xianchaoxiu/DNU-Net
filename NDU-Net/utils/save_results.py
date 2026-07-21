from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from utils.dataset import rate_to_tag


def build_output_paths(
    output_root: str | Path,
    method_name: str,
    dataset_name: str,
    sampling_rate: float,
) -> dict[str, Path]:
    root = Path(output_root)
    rate_dir = f"rate_{rate_to_tag(sampling_rate)}"

    log_dir = root / "logs" / method_name / dataset_name / rate_dir
    image_dir = root / "images" / method_name / dataset_name / rate_dir
    recon_dir = image_dir / "reconstructions"
    vis_dir = image_dir / "visualizations"

    return {
        "root": root,
        "log_dir": log_dir,
        "image_dir": image_dir,
        "recon_dir": recon_dir,
        "vis_dir": vis_dir,
        "metrics_json": log_dir / "metrics.json",
        "per_image_csv": log_dir / "per_image.csv",
        "run_config_json": log_dir / "run_config.json",
        "run_log_txt": log_dir / "run.log",
    }


def ensure_output_paths(paths: dict[str, Path]) -> None:
    for key in ("log_dir", "image_dir", "recon_dir", "vis_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(val) for val in value]
    if isinstance(value, tuple):
        return [_json_safe(val) for val in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
    return value


def save_json(payload: dict[str, Any], save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=True)


def save_csv(rows: list[dict[str, Any]], save_path: str | Path, fieldnames: list[str] | None = None) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if not rows:
            raise ValueError("fieldnames must be provided when saving an empty CSV.")
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def append_log(message: str, save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")
