from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.ndu_net import test as ndu_test
from utils.dataset import compose_rgb_from_y_channel, find_image_paths, load_image, load_y_channel_image
from utils.metrics import FULL_Y_DATA_RANGE, compute_metrics, summarize_records
from utils.save_results import build_output_paths, ensure_output_paths, save_csv, save_json
from utils.visualize import save_image, save_triptych


DEFAULT_TEST_DATA_DIRS = [
    ROOT / "testdata" / "BSD68",
    ROOT / "testdata" / "LIVE29",
    ROOT / "testdata" / "OST300",
    ROOT / "testdata" / "Set11",
    ROOT / "testdata" / "Set14",
    ROOT / "testdata" / "Urban100",
]
DEFAULT_SAMPLING_RATES = [0.25]
DEFAULT_TRANSFORMER_DEPTH = 2


def resolve_checkpoint_dir_for_rate(args: argparse.Namespace, sampling_rate: float) -> Path:
    cs_ratio = ndu_test.sampling_rate_to_cs_ratio(sampling_rate)
    return ndu_test.resolve_rate_checkpoint_dir(Path(args.checkpoint_dir), cs_ratio)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate NDU-Net.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_TEST_DATA_DIRS,
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--sampling-rates", nargs="+", type=float, default=DEFAULT_SAMPLING_RATES)
    parser.add_argument("--sampling-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epoch-num", type=int, default=60)
    parser.add_argument("--use-best", dest="use_best", action="store_true")
    parser.add_argument("--no-use-best", dest="use_best", action="store_false")
    parser.add_argument("--layer-num", type=int, default=ndu_test.DEFAULT_LAYER_NUM)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--group-num", type=int, default=1)
    parser.add_argument("--solver-feature-dim", type=int, default=ndu_test.DEFAULT_SOLVER_FEATURE_DIM)
    parser.add_argument("--prior-feature-channels", type=int, default=ndu_test.DEFAULT_PRIOR_FEATURE_CHANNELS)
    parser.add_argument("--transformer-depth", type=int, default=DEFAULT_TRANSFORMER_DEPTH)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--lambda-newton", type=float, default=0.01)
    parser.add_argument("--coso-gaussian-fraction", type=float, default=0.6)
    parser.add_argument("--coso-filter-channels", type=int, default=8)
    parser.add_argument("--coso-filter-residual-scale", type=float, default=0.1)
    parser.add_argument("--augmentation", dest="augmentation", action="store_true")
    parser.add_argument("--no-augmentation", dest="augmentation", action="store_false")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-save-reconstructions", dest="save_reconstructions", action="store_false")
    parser.add_argument("--no-save-visualizations", dest="save_visualizations", action="store_false")
    parser.set_defaults(
        save_reconstructions=True,
        save_visualizations=True,
        use_best=True,
        augmentation=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sampling_rates = [float(args.sampling_rate)] if args.sampling_rate is not None else [float(rate) for rate in args.sampling_rates]
    data_dirs = [args.data_dir] if args.data_dir is not None else list(args.data_dirs)

    for data_dir in data_dirs:
        image_paths = find_image_paths(data_dir)
        if args.limit and args.limit > 0:
            image_paths = image_paths[: int(args.limit)]
        if not image_paths:
            raise FileNotFoundError(f"No images found under {data_dir}")

        for sampling_rate in sampling_rates:
            checkpoint_dir = resolve_checkpoint_dir_for_rate(args, sampling_rate)
            model = ndu_test.load_model(
                checkpoint_dir=checkpoint_dir,
                sampling_rate=sampling_rate,
                device=args.device,
                extra_args=vars(args),
            )

            dataset_name = Path(data_dir).name
            paths = build_output_paths(
                output_root=args.output_dir,
                method_name="ndu_net",
                dataset_name=dataset_name,
                sampling_rate=sampling_rate,
            )
            ensure_output_paths(paths)
            save_json(
                {
                    "method": "NDU-Net",
                    "data_dir": str(Path(data_dir).resolve()),
                    "checkpoint": str(model.checkpoint_path),
                    "sampling_operator_source": "global_coso_in_checkpoint",
                    "sampling_rate": sampling_rate,
                    "metric_data_range": FULL_Y_DATA_RANGE,
                    "y_range": "full_0_1",
                    "sampling_operator": "global_coso_merged_adjoint_learnable_alpha",
                    "sampling_scope": "full_padded_image",
                    "gaussian_basis": "fixed_orthogonal",
                    "coso_gaussian_fraction": float(args.coso_gaussian_fraction),
                    "coso_filter_channels": int(args.coso_filter_channels),
                    "coso_filter_residual_scale": float(args.coso_filter_residual_scale),
                    "coso_filter_residual_scale_trainable": True,
                    "training_augmentation": bool(args.augmentation),
                    "block_size": int(args.block_size),
                    "device": args.device,
                    "epoch_num": int(args.epoch_num),
                    "use_best": bool(args.use_best),
                    "layer_num": int(args.layer_num),
                    "learning_rate": float(args.learning_rate),
                    "group_num": int(args.group_num),
                    "solver_feature_dim": int(args.solver_feature_dim),
                    "prior_feature_channels": int(args.prior_feature_channels),
                    "transformer_depth": int(args.transformer_depth),
                    "num_heads": int(args.num_heads),
                    "window_size": int(args.window_size),
                    "lambda_newton": float(args.lambda_newton),
                },
                paths["run_config_json"],
            )

            records = []
            for image_path in image_paths:
                image_y = load_y_channel_image(image_path)
                start_time = time.perf_counter()
                reconstruction_y = ndu_test.reconstruct(
                    image=image_y,
                    sampling_rate=sampling_rate,
                    block_size=int(args.block_size),
                    checkpoint_dir=checkpoint_dir,
                    device=args.device,
                    model=model,
                    image_path=image_path,
                    extra_args=vars(args),
                )
                elapsed = time.perf_counter() - start_time
                metrics = compute_metrics(reconstruction_y, image_y)

                recon_path = ""
                vis_path = ""
                reconstruction_rgb = None
                if args.save_reconstructions:
                    reconstruction_rgb = compose_rgb_from_y_channel(reconstruction_y, image_path)
                    recon_file = paths["recon_dir"] / f"{image_path.stem}_recon.png"
                    save_image(reconstruction_rgb, recon_file)
                    recon_path = str(recon_file.resolve())
                if args.save_visualizations:
                    if reconstruction_rgb is None:
                        reconstruction_rgb = compose_rgb_from_y_channel(reconstruction_y, image_path)
                    target_rgb = load_image(image_path, grayscale=False)
                    vis_file = paths["vis_dir"] / f"{image_path.stem}_compare.png"
                    save_triptych(target=target_rgb, prediction=reconstruction_rgb, save_path=vis_file)
                    vis_path = str(vis_file.resolve())

                records.append(
                    {
                        "image_name": image_path.name,
                        "sampling_rate": sampling_rate,
                        "psnr": float(metrics["psnr"]),
                        "ssim": float(metrics["ssim"]),
                        "time_sec": float(elapsed),
                        "reconstruction_path": recon_path,
                        "visualization_path": vis_path,
                    }
                )

            summary = summarize_records(records)
            payload = {
                "method": "NDU-Net",
                "dataset": dataset_name,
                "sampling_rate": sampling_rate,
                "avg_psnr": summary["avg_psnr"],
                "avg_ssim": summary["avg_ssim"],
                "avg_time_sec": summary["avg_time_sec"],
                "num_images": len(records),
            }
            save_csv(records, paths["per_image_csv"])
            save_json(payload, paths["metrics_json"])
            print(json.dumps(payload))


if __name__ == "__main__":
    main()
