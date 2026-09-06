#!/usr/bin/env python3
"""Generate class-balanced synthetic ILD CT slices from real training layouts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffusion_model.unet import create_model
from ild_2d import (
    LABEL_NAMES,
    NUM_CONDITION_CHANNELS,
    GaussianDiffusion2D,
    ILDLayoutDataset2D,
    class_counts,
    discover_records,
    seed_everything,
    training_patient_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=0,
        help=(
            "Desired real+synthetic slice count per class. The default 0 uses "
            "the largest real training-class count."
        ),
    )
    parser.add_argument("--classes", type=int, nargs="+", default=list(LABEL_NAMES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=None,
        help="CFG scale. Default: value stored in checkpoint config, otherwise 2.0",
    )
    parser.add_argument(
        "--blend-mode",
        choices=("roi", "lung", "none"),
        default=None,
        help="Reference blending region. Default: value stored in checkpoint config, otherwise roi",
    )
    parser.add_argument(
        "--inpainting-strength", type=float, default=None,
        help="SDEdit noise strength. Default: checkpoint value, otherwise 0.45",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for sampling")
    invalid = sorted(set(args.classes) - set(LABEL_NAMES))
    if invalid:
        raise ValueError(f"Unknown classes: {invalid}; valid classes are {sorted(LABEL_NAMES)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output is not empty: {args.output_dir}; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    allowed = training_patient_ids(args.split_report)
    records = discover_records(args.data_root, allowed_patients=allowed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]["args"]
    guidance_scale = (
        args.guidance_scale
        if args.guidance_scale is not None
        else float(config.get("guidance_scale", 2.0))
    )
    blend_mode = args.blend_mode or config.get("blend_mode", "roi")
    inpainting_strength = (
        args.inpainting_strength
        if args.inpainting_strength is not None
        else float(config.get("inpainting_strength", 0.45))
    )
    image_size = int(config["image_size"])
    model = create_model(
        image_size,
        int(config["model_channels"]),
        int(config["num_res_blocks"]),
        in_channels=1 + NUM_CONDITION_CHANNELS,
        out_channels=1,
        dims=2,
    ).cuda()
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    diffusion = GaussianDiffusion2D(model, image_size, int(config["timesteps"])).cuda()
    dataset = ILDLayoutDataset2D(records, image_size)
    by_class = {
        label: [index for index, record in enumerate(records) if label in record.labels]
        for label in args.classes
    }
    for label, indexes in by_class.items():
        if not indexes:
            raise ValueError(f"No training layouts contain class {label} ({LABEL_NAMES[label]})")

    seed_everything(args.seed)
    real_counts = class_counts(records)
    target_count = args.target_per_class or max(
        real_counts[label] for label in args.classes
    )
    required = {
        label: max(target_count - real_counts[label], 0) for label in args.classes
    }
    generated_occurrences: Counter[int] = Counter()
    generated = []
    cursor = Counter()
    total_requested = sum(required.values())
    progress = tqdm(total=total_requested, desc="balanced ILD sampling")
    case_index = 0
    while any(generated_occurrences[label] < required[label] for label in args.classes):
        deficits = {
            label: required[label] - generated_occurrences[label]
            for label in args.classes
        }
        target_label = max(deficits, key=deficits.get)
        indexes = by_class[target_label]
        source_index = indexes[cursor[target_label] % len(indexes)]
        cursor[target_label] += 1
        item = dataset[source_index]
        condition = item["condition"].unsqueeze(0).cuda()
        reference = item["image"].unsqueeze(0).cuda()
        lung = item["lung"].unsqueeze(0).cuda()
        synthetic = diffusion.sample(
            condition,
            reference=reference,
            lung=lung,
            guidance_scale=guidance_scale,
            blend_mode=blend_mode,
            inpainting_strength=inpainting_strength,
        )
        image = ((synthetic[0, 0].cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        source_record = records[source_index]
        roi = np.load(source_record.roi_path).astype(np.uint8)
        lung_np = (np.load(source_record.lung_path) > 0).astype(np.uint8)
        if image.shape != roi.shape:
            image = np.asarray(
                Image.fromarray(image).resize(
                    (roi.shape[1], roi.shape[0]), resample=Image.Resampling.BILINEAR
                )
            )

        case_name = f"synthetic_{case_index:06d}"
        case_dir = args.output_dir / case_name
        for folder in ("images", "lung_masks", "roi_masks"):
            (case_dir / folder).mkdir(parents=True, exist_ok=True)
        np.save(case_dir / "images" / "slice_1.npy", image)
        np.save(case_dir / "lung_masks" / "slice_1.npy", lung_np)
        np.save(case_dir / "roi_masks" / "slice_1.npy", roi)
        save_png(image, case_dir / "images" / "slice_1.png")
        save_png(lung_np * 255, case_dir / "lung_masks" / "slice_1.png")
        roi_display = np.rint(roi.astype(np.float32) / max(int(roi.max()), 1) * 255)
        save_png(roi_display, case_dir / "roi_masks" / "slice_1.png")
        labels = sorted(int(value) for value in np.unique(roi) if int(value) in LABEL_NAMES)
        for label in labels:
            if label in args.classes and generated_occurrences[label] < required[label]:
                generated_occurrences[label] += 1
                progress.update(1)
        generated.append(
            {
                "case": case_name,
                "source_patient": item["patient_id"],
                "source_slice": item["slice_id"],
                "labels": labels,
                "target_class": target_label,
                "seed": args.seed + case_index,
            }
        )
        case_index += 1
        torch.manual_seed(args.seed + case_index)
        torch.cuda.manual_seed_all(args.seed + case_index)
    progress.close()
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "guidance_scale": guidance_scale,
        "blend_mode": blend_mode,
        "inpainting_strength": inpainting_strength,
        "real_training_class_counts": real_counts,
        "requested_classes": {str(label): LABEL_NAMES[label] for label in args.classes},
        "target_total_count_per_class": target_count,
        "required_synthetic_occurrences": required,
        "generated_class_occurrences": dict(generated_occurrences),
        "num_synthetic_cases": len(generated),
        "cases": generated,
    }
    (args.output_dir / "generation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
