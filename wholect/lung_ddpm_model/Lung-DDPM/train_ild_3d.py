#!/usr/bin/env python3
"""Train the first 3-D ILD Lung-DDPM with periodic volumetric inference."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion_model.unet import create_model
from ild_3d import (
    NUM_CONDITION_CHANNELS_3D, GaussianDiffusion3D, ILDVolumeDataset3D,
    label_balanced_sampler, records_from_nnunet_report, require_prepared,
    seed_everything, update_ema,
)
from ild_3d_evaluation import save_3d_inference_preview


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="nnUNet_raw/Dataset503_ILD3DROI (used for patient splits)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--model-channels", type=int, default=32)
    parser.add_argument("--num-res-blocks", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--outside-loss-weight", type=float, default=0.1)
    parser.add_argument("--lung-loss-weight", type=float, default=1.0)
    parser.add_argument("--roi-loss-weight", type=float, default=8.0)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--preview-every", type=int, default=10000)
    parser.add_argument("--preview-strength", type=float, default=1.0,
                        help="1.0 is pure-noise generation; lower is SDEdit reconstruction")
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = args_parser()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if not 0 < args.preview_strength <= 1: raise ValueError("preview strength must be in (0,1]")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_records = records_from_nnunet_report(args.dataset_dir, "train")
    val_records = records_from_nnunet_report(args.dataset_dir, "val")
    require_prepared(train_records + val_records)
    train_dataset = ILDVolumeDataset3D(train_records, augment=True)
    val_dataset = ILDVolumeDataset3D(val_records, augment=False)
    sampler = label_balanced_sampler(train_records, args.seed, args.samples_per_epoch)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    device = torch.device("cuda")
    model = create_model(
        args.volume_size, args.model_channels, args.num_res_blocks,
        in_channels=1 + NUM_CONDITION_CHANNELS_3D, out_channels=1,
        # The upstream custom checkpoint function recomputes outside autocast
        # and is incompatible with AMP (Half input versus Float bias).
        dims=3, use_checkpoint=False,
    ).to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion3D(model, args.volume_size, args.timesteps).to(device)
    ema_diffusion = GaussianDiffusion3D(ema_model, args.volume_size, args.timesteps).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"]); ema_model.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"]); scaler.load_state_dict(checkpoint["scaler"])
        step = int(checkpoint["step"])
    config = {
        "train_volumes": len(train_records), "validation_volumes": len(val_records),
        "condition_channels": ["lung"] + [f"roi_label_{x:02d}" for x in range(1, 18)],
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output_dir / "training_config.json").write_text(json.dumps(config, indent=2) + "\n")
    log_path = args.output_dir / "training_loss.csv"
    if not log_path.exists() or step == 0:
        with log_path.open("w", newline="") as handle:
            csv.writer(handle).writerow(("step", "weighted_l1_noise_loss"))
    iterator, progress = iter(loader), tqdm(total=args.steps, initial=step, desc="ILD 3D DDPM")
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True); accumulated = 0.0
        for _ in range(args.gradient_accumulation):
            try: batch = next(iterator)
            except StopIteration: iterator = iter(loader); batch = next(iterator)
            image = batch["image"].to(device, non_blocking=True)
            condition = batch["condition"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=not args.no_amp):
                loss = diffusion(
                    image, condition, args.outside_loss_weight, args.lung_loss_weight,
                    args.roi_loss_weight, args.condition_dropout,
                ) / args.gradient_accumulation
            scaler.scale(loss).backward(); accumulated += float(loss.detach())
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); update_ema(ema_model, model, args.ema_decay)
        step += 1; progress.update(1); progress.set_postfix(loss=f"{accumulated:.4f}")
        if step % 100 == 0:
            with log_path.open("a", newline="") as handle: csv.writer(handle).writerow((step, accumulated))
        should_save = step % args.save_every == 0 or step == args.steps
        if should_save:
            checkpoint_path = args.output_dir / f"ild_ddpm_3d_step_{step:07d}.pth"
            torch.save({"step": step, "model": model.state_dict(), "ema": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "config": config}, checkpoint_path)
            progress.write(f"Saved checkpoint: {checkpoint_path}")
        if step % args.preview_every == 0 or step == args.steps:
            preview, nifti, metrics = save_3d_inference_preview(
                ema_diffusion, val_dataset, 0, args.output_dir, step, device,
                args.seed, args.preview_strength, args.guidance_scale,
            )
            progress.write(f"Saved 3-D inference preview: {preview}")
            progress.write(f"Saved generated NIfTI: {nifti}")
            progress.write(f"Saved preview metrics: {metrics}")
    progress.close()


if __name__ == "__main__":
    main()
