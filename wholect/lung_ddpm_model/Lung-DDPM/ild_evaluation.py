"""Validation previews and condition-control diagnostics for ILD DDPMs."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from ild_2d import GaussianDiffusion2D, ILDLayoutDataset2D, LABEL_NAMES


CONDITION_COLORS = np.asarray(
    [
        (0, 0, 0),
        (0, 200, 0),
        (255, 215, 0),
        (0, 220, 255),
        (255, 45, 45),
        (255, 0, 255),
        (40, 90, 255),
    ],
    dtype=np.uint8,
)


def select_preview_indices(records, count: int) -> list[int]:
    """Select deterministic, interpretable ROI-positive validation examples."""
    if count <= 0:
        return []
    selected: list[int] = []
    covered: set[int] = set()
    for label in sorted(LABEL_NAMES):
        index = next(
            (i for i, record in enumerate(records) if record.labels == (label,)),
            None,
        )
        if index is not None:
            selected.append(index)
            covered.add(label)
            if len(selected) == count:
                return selected
    for index, record in enumerate(records):
        if record.labels and any(label not in covered for label in record.labels):
            selected.append(index)
            covered.update(record.labels)
            if len(selected) == count:
                return selected
    for index, record in enumerate(records):
        if record.labels and index not in selected:
            selected.append(index)
            if len(selected) == count:
                break
    return selected


def masked_metrics(
    real: np.ndarray, generated: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | None]:
    """Compute paired reconstruction diagnostics within a binary mask."""
    if not np.any(mask):
        return {"pixels": 0, "MAE": None, "PSNR": None, "SSIM": None}
    difference = real.astype(np.float32) - generated.astype(np.float32)
    mse = float(np.mean(difference[mask] ** 2))
    _, ssim_map = structural_similarity(
        real.astype(np.float32), generated.astype(np.float32), data_range=255, full=True
    )
    return {
        "pixels": int(mask.sum()),
        "MAE": float(np.mean(np.abs(difference[mask]))),
        "PSNR": float(10 * math.log10(255**2 / mse)) if mse else None,
        "SSIM": float(np.mean(ssim_map[mask])),
    }


def _u8(images: torch.Tensor) -> np.ndarray:
    return ((images.detach().cpu().numpy()[:, 0] + 1) * 127.5).round().clip(0, 255).astype(np.uint8)


def _condition_layout(roi: np.ndarray) -> np.ndarray:
    layout = np.zeros((*roi.shape, 3), dtype=np.uint8)
    for label in LABEL_NAMES:
        layout[roi == label] = CONDITION_COLORS[label]
    return layout


def _layout_from_condition(condition: torch.Tensor) -> np.ndarray:
    """Convert six disease channels to the fixed-color layout used in previews."""
    disease = condition[1:].detach().cpu().numpy()
    active = disease.max(axis=0) > 0.5
    roi = np.zeros(active.shape, dtype=np.uint8)
    roi[active] = disease.argmax(axis=0)[active].astype(np.uint8) + 1
    return _condition_layout(roi)


def condition_outline(image: np.ndarray, layout: np.ndarray) -> np.ndarray:
    """Draw colored condition contours without hiding CT texture."""
    result = np.repeat(image[..., None], 3, axis=2)
    mask = np.any(layout != 0, axis=2)
    interior = mask.copy()
    interior[1:-1, 1:-1] &= (
        mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    )
    edge = mask & ~interior
    thick_edge = edge.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        thick_edge |= np.roll(edge, shift=(dy, dx), axis=(0, 1))
    result[thick_edge & mask] = layout[thick_edge & mask]
    return result


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    lines: list[str],
    font: ImageFont.ImageFont,
    spacing: int = 17,
) -> None:
    x, y = xy
    for line in lines:
        draw.text((x, y), line, fill="black", font=font)
        y += spacing


def _sample_with_seed(
    diffusion: GaussianDiffusion2D,
    condition: torch.Tensor,
    reference: torch.Tensor,
    lung: torch.Tensor,
    seed: int,
    guidance_scale: float,
    blend_mode: str,
    inpainting_strength: float,
) -> torch.Tensor:
    device_index = reference.device.index
    devices = [torch.cuda.current_device() if device_index is None else device_index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return diffusion.sample(
            condition,
            reference=reference,
            lung=lung,
            guidance_scale=guidance_scale,
            blend_mode=blend_mode,
            inpainting_strength=inpainting_strength,
        )


def save_reconstruction_preview(
    ema_diffusion: GaussianDiffusion2D,
    dataset: ILDLayoutDataset2D,
    indices: list[int],
    output_dir: Path,
    step: int,
    seed: int,
    device: torch.device,
    guidance_scale: float,
    blend_mode: str,
    inpainting_strength: float,
) -> tuple[Path, Path]:
    """Save compact paired diagnostics; these are not realism scores."""
    if not indices:
        raise ValueError("At least one preview index is required")
    items = [dataset[index] for index in indices]
    reference = torch.stack([item["image"] for item in items]).to(device)
    condition = torch.stack([item["condition"] for item in items]).to(device)
    lung = torch.stack([item["lung"] for item in items]).to(device)
    synthetic = _sample_with_seed(
        ema_diffusion, condition, reference, lung, seed,
        guidance_scale, blend_mode, inpainting_strength,
    )

    real_u8, synthetic_u8 = _u8(reference), _u8(synthetic)
    size, header, gap = real_u8.shape[-1], 72, 8
    canvas = Image.new("RGB", (size * 3 + gap * 2, (size + header) * len(items)), "white")
    draw, font = ImageDraw.Draw(canvas), _font(max(10, min(13, size // 20)))
    metric_rows = []
    for row, (item, real, generated) in enumerate(zip(items, real_u8, synthetic_u8)):
        top = row * (size + header)
        labels = dataset.records[indices[row]].labels
        roi = item["roi"].numpy()
        roi_mask = np.isin(roi, list(LABEL_NAMES))
        lung_mask = item["lung"].numpy()[0] > 0.5
        metrics = {
            "patient_id": str(item["patient_id"]),
            "slice_id": str(item["slice_id"]),
            "labels": list(labels),
            "roi_fraction": float(roi_mask.mean()),
            "lung": masked_metrics(real, generated, lung_mask),
            "roi": masked_metrics(real, generated, roi_mask),
        }
        metric_rows.append(metrics)
        difference = np.clip(
            np.abs(real.astype(np.int16) - generated.astype(np.int16)) * 4, 0, 255
        ).astype(np.uint8)
        roi_metrics = metrics["roi"]
        p = "NA" if roi_metrics["PSNR"] is None else f"{roi_metrics['PSNR']:.2f}"
        s = "NA" if roi_metrics["SSIM"] is None else f"{roi_metrics['SSIM']:.3f}"
        _draw_lines(draw, (8, top + 5), ["Validation reference", f"labels={list(labels)}"], font)
        _draw_lines(
            draw, (size + gap + 8, top + 5),
            ["EMA reconstruction", f"step={step}", f"ROI: PSNR={p}  SSIM={s}"], font,
        )
        _draw_lines(
            draw, (size * 2 + gap * 2 + 8, top + 5),
            ["Absolute difference", "display scale: 4x"], font,
        )
        layout = _condition_layout(roi)
        canvas.paste(Image.fromarray(condition_outline(real, layout)), (0, top + header))
        canvas.paste(Image.fromarray(condition_outline(generated, layout)), (size + gap, top + header))
        canvas.paste(
            Image.fromarray(difference, mode="L").convert("RGB"),
            (size * 2 + gap * 2, top + header),
        )

    preview_dir = output_dir / "previews"
    metrics_dir = output_dir / "preview_metrics"
    preview_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = preview_dir / f"validation_real_vs_diffusion_step_{step:07d}.png"
    metric_path = metrics_dir / f"metrics_step_{step:07d}.json"
    canvas.save(path)
    metric_path.write_text(json.dumps({
        "step": step,
        "split": "val",
        "inpainting_strength": inpainting_strength,
        "metric_interpretation": (
            "Paired reference-reconstruction diagnostics only; not standalone realism, "
            "diversity, condition-fidelity, or medical-validity scores."
        ),
        "cases": metric_rows,
    }, indent=2) + "\n", encoding="utf-8")
    return path, metric_path


def _shift_no_wrap(array: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    shifted = torch.zeros_like(array)
    height, width = array.shape[-2:]
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    dst_y0, dst_y1 = max(0, dy), min(height, height + dy)
    dst_x0, dst_x1 = max(0, dx), min(width, width + dx)
    shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = array[..., src_y0:src_y1, src_x0:src_x1]
    return shifted


def moved_disease_condition(condition: torch.Tensor) -> torch.Tensor:
    """Move disease layouts while retaining the largest non-overlapping lung area."""
    result = condition.clone()
    disease, lung = condition[:, 1:], condition[:, :1]
    distance = max(8, condition.shape[-1] // 5)
    candidates = ((0, distance), (0, -distance), (distance, 0), (-distance, 0))
    for batch_index in range(condition.shape[0]):
        original = disease[batch_index : batch_index + 1]
        original_union = original.sum(1, keepdim=True) > 0
        best, best_score = original, -1.0
        for dy, dx in candidates:
            candidate = _shift_no_wrap(original, dy, dx) * lung[batch_index : batch_index + 1]
            union = candidate.sum(1, keepdim=True) > 0
            retained = float(union.sum())
            overlap = float((union & original_union).sum())
            score = retained - overlap
            if score > best_score:
                best, best_score = candidate, score
        result[batch_index : batch_index + 1, 1:] = best
    return result


def _masked_difference(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    if not np.any(mask):
        return {"pixels": 0, "MAE": 0.0, "changed_fraction_gt_5": 0.0}
    difference = np.abs(a.astype(np.float32) - b.astype(np.float32))[mask]
    return {
        "pixels": int(mask.sum()),
        "MAE": float(difference.mean()),
        "changed_fraction_gt_5": float((difference > 5).mean()),
    }


@torch.no_grad()
def save_condition_control_evaluation(
    ema_diffusion: GaussianDiffusion2D,
    dataset: ILDLayoutDataset2D,
    indices: list[int],
    output_dir: Path,
    step: int,
    seed: int,
    num_seeds: int,
    device: torch.device,
    guidance_scale: float,
    blend_mode: str,
    inpainting_strength: float,
) -> tuple[Path, Path]:
    """Measure whether outputs react to label removal, swapping, movement and seed."""
    if num_seeds < 2:
        raise ValueError("num_seeds must be at least 2 for diversity diagnostics")
    items = [dataset[index] for index in indices]
    reference = torch.stack([item["image"] for item in items]).to(device)
    condition = torch.stack([item["condition"] for item in items]).to(device)
    lung = torch.stack([item["lung"] for item in items]).to(device)
    null_condition = condition.clone()
    null_condition[:, 1:] = 0
    swapped_condition = condition.clone()
    swapped_condition[:, 1:] = torch.roll(condition[:, 1:], shifts=1, dims=1)
    moved_condition = moved_disease_condition(condition)

    variants = {
        "correct": condition,
        "no_disease": null_condition,
        "label_swap": swapped_condition,
        "roi_moved": moved_condition,
    }
    generated = {
        name: _sample_with_seed(
            ema_diffusion, variant, reference, lung, seed,
            guidance_scale, blend_mode, inpainting_strength,
        )
        for name, variant in variants.items()
    }
    seed_samples = [generated["correct"]]
    for offset in range(1, num_seeds):
        seed_samples.append(_sample_with_seed(
            ema_diffusion, condition, reference, lung, seed + offset,
            guidance_scale, blend_mode, inpainting_strength,
        ))

    reference_u8 = _u8(reference)
    variant_u8 = {name: _u8(value) for name, value in generated.items()}
    seed_u8 = [_u8(value) for value in seed_samples]
    size, header, gap = reference_u8.shape[-1], 48, 8
    columns = ["Reference", "Correct condition", "No disease", "Label swap", "ROI moved", "|Correct - null| (4x)"]
    canvas = Image.new("RGB", (size * len(columns) + gap * (len(columns) - 1), header + size * len(items)), "white")
    draw, font = ImageDraw.Draw(canvas), _font(max(10, min(13, size // 20)))
    for column, title in enumerate(columns):
        _draw_lines(draw, (column * (size + gap) + 7, 7), [title], font)

    rows = []
    for row, item in enumerate(items):
        top = header + row * size
        roi = item["roi"].numpy()
        labels = list(dataset.records[indices[row]].labels)
        roi_mask = np.isin(roi, list(LABEL_NAMES))
        lung_mask = item["lung"].numpy()[0] > 0.5
        real = reference_u8[row]
        correct = variant_u8["correct"][row]
        null = variant_u8["no_disease"][row]
        swap = variant_u8["label_swap"][row]
        moved = variant_u8["roi_moved"][row]
        difference = np.clip(np.abs(correct.astype(np.int16) - null.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        layout = _condition_layout(roi)
        swapped_layout = _layout_from_condition(swapped_condition[row])
        moved_layout = _layout_from_condition(moved_condition[row])
        panels = [
            condition_outline(real, layout),
            condition_outline(correct, layout),
            np.repeat(null[..., None], 3, axis=2),
            condition_outline(swap, swapped_layout),
            condition_outline(moved, moved_layout),
            np.repeat(difference[..., None], 3, axis=2),
        ]
        for column, panel in enumerate(panels):
            canvas.paste(Image.fromarray(panel), (column * (size + gap), top))
        row_label = f"labels={labels}"
        label_box = draw.textbbox((0, 0), row_label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        draw.rectangle(
            (4, top + 4, 12 + label_width, top + 12 + label_height),
            fill="white",
        )
        draw.text((8, top + 7), row_label, fill="black", font=font)

        pairwise_diversity = []
        for first, second in itertools.combinations(seed_u8, 2):
            pairwise_diversity.append(_masked_difference(first[row], second[row], lung_mask)["MAE"])
        rows.append({
            "patient_id": str(item["patient_id"]),
            "slice_id": str(item["slice_id"]),
            "labels": labels,
            "correct_vs_no_disease_roi": _masked_difference(correct, null, roi_mask),
            "correct_vs_no_disease_lung": _masked_difference(correct, null, lung_mask),
            "correct_vs_label_swap_roi": _masked_difference(correct, swap, roi_mask),
            "correct_vs_roi_moved_lung": _masked_difference(correct, moved, lung_mask),
            "mean_pairwise_seed_mae_in_lung": float(np.mean(pairwise_diversity)),
        })

    control_dir = output_dir / "condition_control"
    control_dir.mkdir(parents=True, exist_ok=True)
    image_path = control_dir / f"condition_control_step_{step:07d}.png"
    metric_path = control_dir / f"condition_control_step_{step:07d}.json"
    canvas.save(image_path)
    metric_path.write_text(json.dumps({
        "step": step,
        "seed": seed,
        "num_diversity_seeds": num_seeds,
        "interpretation": {
            "condition_effect": "Correct and ablated/swapped/moved samples share the same stochastic seed. Near-zero differences indicate condition ignoring.",
            "diversity": "Pairwise lung MAE across seeds detects stochastic variation, not realism.",
            "medical_validity": "These diagnostics do not establish that generated disease texture is medically correct.",
        },
        "cases": rows,
    }, indent=2) + "\n", encoding="utf-8")
    return image_path, metric_path
