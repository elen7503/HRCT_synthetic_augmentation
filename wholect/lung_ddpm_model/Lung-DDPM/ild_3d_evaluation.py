"""Checkpoint inference previews for the 3-D ILD diffusion model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch


def to_hu(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy()[0, 0]
    return np.rint((value + 1) * 700 - 1000).astype(np.int16)


def center_indices(roi: np.ndarray, lung: np.ndarray):
    coords = np.argwhere(roi > 0)
    if not len(coords): coords = np.argwhere(lung > 0)
    return np.rint(coords.mean(0)).astype(int)


def get_plane(array, center, plane):
    z, y, x = center
    return (array[z], array[:, y], array[:, :, x])[plane]


@torch.no_grad()
def save_3d_inference_preview(diffusion, dataset, index: int, output_dir: Path,
                              step: int, device: torch.device, seed: int,
                              strength: float, guidance_scale: float):
    item = dataset[index]
    reference = item["image"].unsqueeze(0).to(device)
    condition = item["condition"].unsqueeze(0).to(device)
    lung = item["lung"].unsqueeze(0).to(device)
    with torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        generated = diffusion.sample(condition, reference, lung, strength, guidance_scale)
    real, synthetic = to_hu(reference), to_hu(generated)
    lung_np = item["lung"].numpy()[0] > 0
    roi = item["roi"].numpy()
    center = center_indices(roi, lung_np)
    difference = np.abs(real.astype(np.float32) - synthetic.astype(np.float32))
    condition_view = np.where(roi > 0, roi, np.where(lung_np, 0.25, 0))
    titles = ("Reference CT", "EMA generated CT", "Condition", "Absolute difference")
    volumes = (real, synthetic, condition_view, difference)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for plane, plane_name in enumerate(("Axial", "Coronal", "Sagittal")):
        for column, (title, volume) in enumerate(zip(titles, volumes)):
            ax = axes[plane, column]
            image = get_plane(volume, center, plane)
            if column < 2:
                ax.imshow(image, cmap="gray", vmin=-1000, vmax=400)
                roi_plane = get_plane(roi, center, plane)
                if np.any(roi_plane): ax.contour(roi_plane, levels=[0.5], colors="red", linewidths=.8)
            elif column == 2:
                ax.imshow(image, cmap="turbo", vmin=0, vmax=17)
            else:
                ax.imshow(image, cmap="magma", vmin=0, vmax=400)
            ax.set_title(f"{plane_name} — {title}"); ax.axis("off")
    fig.suptitle(f"step={step} | case={item['case_id']} | strength={strength} | seed={seed}")
    preview_dir, nifti_dir = output_dir / "previews_3d", output_dir / "inference_nifti"
    preview_dir.mkdir(parents=True, exist_ok=True); nifti_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"inference_step_{step:07d}.png"
    fig.savefig(preview_path, dpi=130); plt.close(fig)
    affine = item["affine"].numpy()
    synthetic_xyz = np.transpose(synthetic, (2, 1, 0))
    nifti_path = nifti_dir / f"{item['case_id']}_step_{step:07d}.nii.gz"
    nib.save(nib.Nifti1Image(synthetic_xyz, affine), nifti_path)
    roi_mask = roi > 0
    metrics = {
        "step": step, "case_id": item["case_id"], "patient_id": item["patient_id"],
        "seed": seed, "strength": strength,
        "lung_mae_hu": float(difference[lung_np].mean()),
        "roi_mae_hu": float(difference[roi_mask].mean()) if roi_mask.any() else None,
        "interpretation": "Paired reference/inference diagnostics; not a realism or medical-validity score.",
    }
    metric_path = preview_dir / f"inference_step_{step:07d}.json"
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return preview_path, nifti_path, metric_path
