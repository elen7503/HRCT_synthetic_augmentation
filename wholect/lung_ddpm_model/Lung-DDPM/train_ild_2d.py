#!/usr/bin/env python3
"""Train a class-balanced 2-D conditional DDPM for ILD CT synthesis."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_model.unet import create_model
from ild_evaluation import (
    save_condition_control_evaluation,
    save_reconstruction_preview,
    select_preview_indices,
)
from ild_2d import (
    LABEL_NAMES,
    NUM_CONDITION_CHANNELS,
    GaussianDiffusion2D,
    ILDLayoutDataset2D,
    class_counts,
    dataset_manifest,
    discover_records,
    mixed_condition_sampler,
    seed_everything,
    split_patient_ids,
    training_patient_ids,
    update_ema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--split-report",
        type=Path,
        required=True,
        help="conversion_report.json used to restrict training to real train patients",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--num-res-blocks", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument(
        "--roi-sample-fraction",
        type=float,
        default=0.5,
        help=(
            "Expected fraction of sampled training items with a labelled disease ROI. "
            "The remainder are CT+lung null-condition samples; they are unknown, not healthy."
        ),
    )
    parser.add_argument(
        "--foreground-only",
        action="store_true",
        help="Legacy ablation: discard all null-condition CT+lung samples.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--outside-loss-weight", type=float, default=0.25)
    parser.add_argument("--lung-loss-weight", type=float, default=1.0)
    parser.add_argument("--roi-loss-weight", type=float, default=5.0)
    parser.add_argument(
        "--class-loss-weights",
        type=float,
        nargs=6,
        default=(1.0, 2.0, 2.0, 1.0, 2.0, 3.0),
        metavar=("HEALTHY", "EMPHYSEMA", "GGO", "FIBROSIS", "MICRONODULES", "CONSOLIDATION"),
    )
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument(
        "--max-training-strength", type=float, default=1.0,
        help=(
            "Restrict training timesteps to this fraction of the diffusion trajectory. "
            "Use about 0.65 for small-data image-to-image training."
        ),
    )
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--blend-mode", choices=("roi", "lung", "none"), default="roi")
    parser.add_argument(
        "--inpainting-strength", type=float, default=0.45,
        help="SDEdit noise strength in (0, 1]; lower values preserve more real ROI anatomy",
    )
    parser.add_argument("--roi-crop-probability", type=float, default=0.5)
    parser.add_argument("--roi-crop-size", type=int, default=384)
    parser.add_argument("--rotation-degrees", type=float, default=10.0)
    parser.add_argument("--scale-range", type=float, nargs=2, default=(0.9, 1.1))
    parser.add_argument("--intensity-jitter", type=float, default=0.05)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--preview-count",
        type=int,
        default=1,
        help="Number of fixed real/synthetic pairs saved whenever a checkpoint is written",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not generate checkpoint preview images",
    )
    parser.add_argument(
        "--control-eval-every",
        type=int,
        default=10000,
        help="Run label ablation/swap/ROI-move/multi-seed diagnostics every N steps; 0 disables.",
    )
    parser.add_argument(
        "--control-seeds",
        type=int,
        default=3,
        help="Number of stochastic seeds used by condition-control diversity diagnostics.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Load --resume and generate its validation preview/metrics without training",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training")
    seed_everything(args.seed)
    if not 0 <= args.roi_sample_fraction <= 1:
        raise ValueError("--roi-sample-fraction must be between 0 and 1")
    if args.control_eval_every < 0:
        raise ValueError("--control-eval-every cannot be negative")
    if args.control_eval_every and args.control_seeds < 2:
        raise ValueError("--control-seeds must be at least 2 when control evaluation is enabled")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed = training_patient_ids(args.split_report)
    records = discover_records(
        args.data_root,
        allowed_patients=allowed,
        require_foreground=args.foreground_only,
    )
    roi_records = [record for record in records if record.has_disease_condition]
    null_records = [record for record in records if not record.has_disease_condition]
    dataset = ILDLayoutDataset2D(
        records,
        args.image_size,
        augment=not args.no_augmentation,
        roi_crop_probability=args.roi_crop_probability,
        roi_crop_size=args.roi_crop_size,
        rotation_degrees=args.rotation_degrees,
        scale_range=tuple(args.scale_range),
        intensity_jitter=args.intensity_jitter,
    )
    validation_patients = split_patient_ids(args.split_report, "val")
    validation_all_records = discover_records(
        args.data_root, allowed_patients=validation_patients, require_foreground=False
    )
    validation_records = [
        record for record in validation_all_records if record.has_disease_condition
    ]
    if not validation_records:
        raise ValueError("Validation split has no class-labelled ROI slices for diagnostics")
    validation_dataset = ILDLayoutDataset2D(validation_records, args.image_size)
    train_manifest = dataset_manifest(records, args.data_root)
    validation_manifest = dataset_manifest(validation_all_records, args.data_root)
    sampler = mixed_condition_sampler(
        records,
        args.seed,
        args.samples_per_epoch,
        roi_sample_fraction=1.0 if args.foreground_only else args.roi_sample_fraction,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    device = torch.device("cuda")
    model = create_model(
        args.image_size,
        args.model_channels,
        args.num_res_blocks,
        in_channels=1 + NUM_CONDITION_CHANNELS,
        out_channels=1,
        dims=2,
    ).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion2D(model, args.image_size, args.timesteps).to(device)
    ema_diffusion = GaussianDiffusion2D(
        ema_model, args.image_size, args.timesteps
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        checkpoint_manifest = checkpoint.get("config", {}).get("train_manifest_sha256")
        if checkpoint_manifest is None:
            raise RuntimeError(
                "The checkpoint predates dataset manifests and cannot be safely resumed. "
                "Start a new output directory, or use it for preview/sampling only."
            )
        if checkpoint_manifest != train_manifest["sha256"]:
            raise RuntimeError(
                "Training data do not match the checkpoint manifest: "
                f"checkpoint={checkpoint_manifest}, current={train_manifest['sha256']}"
            )
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])

    metadata = {
        "data_root": str(args.data_root.resolve()),
        "split_report": str(args.split_report.resolve()),
        "num_records": len(records),
        "num_roi_condition_records": len(roi_records),
        "num_null_unknown_records": len(null_records),
        "null_condition_semantics": (
            "Disease channels are zero because no class-labelled ROI is available. "
            "This does not assert that the slice is healthy or lesion-free."
        ),
        "num_train_patients": len(allowed),
        "num_validation_records": len(validation_all_records),
        "num_validation_preview_records": len(validation_records),
        "num_validation_preview_patients": len(validation_patients),
        "class_slice_counts": class_counts(records),
        "train_manifest_sha256": train_manifest["sha256"],
        "validation_manifest_sha256": validation_manifest["sha256"],
        "condition_channels": [
            "lung",
            "healthy_control",
            "emphysema",
            "ground_glass",
            "fibrosis",
            "micronodules",
            "consolidation",
        ],
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    if not args.preview_only:
        (args.output_dir / "training_config.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "train_manifest.json").write_text(
            json.dumps(train_manifest, indent=2) + "\n", encoding="utf-8"
        )
        (args.output_dir / "validation_manifest.json").write_text(
            json.dumps(validation_manifest, indent=2) + "\n", encoding="utf-8"
        )
    preview_indices = select_preview_indices(validation_records, max(args.preview_count, 0))
    if args.preview_only:
        if args.resume is None:
            raise ValueError("--preview-only requires --resume CHECKPOINT")
        if not preview_indices:
            raise ValueError("--preview-only requires --preview-count greater than zero")
        preview_path, metric_path = save_reconstruction_preview(
            ema_diffusion,
            validation_dataset,
            preview_indices,
            args.output_dir,
            step,
            args.seed,
            device,
            args.guidance_scale,
            args.blend_mode,
            args.inpainting_strength,
        )
        print(f"Saved checkpoint preview: {preview_path}")
        print(f"Saved checkpoint metrics: {metric_path}")
        if args.control_eval_every:
            control_path, control_metrics = save_condition_control_evaluation(
                ema_diffusion,
                validation_dataset,
                preview_indices,
                args.output_dir,
                step,
                args.seed,
                args.control_seeds,
                device,
                args.guidance_scale,
                args.blend_mode,
                args.inpainting_strength,
            )
            print(f"Saved condition-control preview: {control_path}")
            print(f"Saved condition-control metrics: {control_metrics}")
        return
    iterator = iter(loader)
    loss_log_path = args.output_dir / "training_loss.csv"
    if step == 0 or not loss_log_path.exists():
        with loss_log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                ("step", "mean_training_loss", "sampled_roi_condition_fraction")
            )
    loss_sum = 0.0
    loss_count = 0
    roi_sample_sum = 0
    progress = tqdm(total=args.steps, initial=step, desc="ILD DDPM training")
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        image = batch["image"].to(device, non_blocking=True)
        condition = batch["condition"].to(device, non_blocking=True)
        roi_sample_sum += int(batch["has_disease_condition"].sum())
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion(
            image,
            condition,
            outside_weight=args.outside_loss_weight,
            lung_weight=args.lung_loss_weight,
            roi_weight=args.roi_loss_weight,
            class_weights=tuple(args.class_loss_weights),
            condition_dropout=args.condition_dropout,
            max_training_strength=args.max_training_strength,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(ema_model, model, args.ema_decay)
        step += 1
        loss_sum += float(loss.item())
        loss_count += 1
        progress.update(1)
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if step % args.log_every == 0:
            with loss_log_path.open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(
                    (
                        step,
                        loss_sum / max(loss_count, 1),
                        roi_sample_sum / max(loss_count * args.batch_size, 1),
                    )
                )
            loss_sum = 0.0
            loss_count = 0
            roi_sample_sum = 0
        if step % args.save_every == 0 or step == args.steps:
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": metadata,
                },
                args.output_dir / f"ild_ddpm_2d_step_{step:07d}.pth",
            )
            if not args.no_preview and preview_indices:
                preview_path, metric_path = save_reconstruction_preview(
                    ema_diffusion,
                    validation_dataset,
                    preview_indices,
                    args.output_dir,
                    step,
                    args.seed,
                    device,
                    args.guidance_scale,
                    args.blend_mode,
                    args.inpainting_strength,
                )
                progress.write(f"Saved checkpoint preview: {preview_path}")
                progress.write(f"Saved checkpoint metrics: {metric_path}")
                if args.control_eval_every and step % args.control_eval_every == 0:
                    control_path, control_metrics = save_condition_control_evaluation(
                        ema_diffusion,
                        validation_dataset,
                        preview_indices,
                        args.output_dir,
                        step,
                        args.seed,
                        args.control_seeds,
                        device,
                        args.guidance_scale,
                        args.blend_mode,
                        args.inpainting_strength,
                    )
                    progress.write(f"Saved condition-control preview: {control_path}")
                    progress.write(f"Saved condition-control metrics: {control_metrics}")
    progress.close()


if __name__ == "__main__":
    main()
