"""Dataset, balanced sampling and diffusion core for 3-D ILD synthesis."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from ild_2d import cosine_beta_schedule, extract, update_ema, seed_everything


NUM_ROI_CLASSES = 17
NUM_CONDITION_CHANNELS_3D = 1 + NUM_ROI_CLASSES


@dataclass(frozen=True)
class VolumeRecord:
    case_id: str
    patient_id: str
    split: str
    ct_path: Path
    lung_path: Path
    roi_path: Path
    labels: tuple[int, ...]


def records_from_nnunet_report(dataset_dir: Path, split: str) -> list[VolumeRecord]:
    """Use Dataset503's patient-level assignment and original source folders."""
    dataset_dir = Path(dataset_dir)
    report = json.loads((dataset_dir / "conversion_report.json").read_text())
    records = []
    for case_id, row in report["case_mapping"].items():
        if row["split"] != split:
            continue
        source = Path(report["source"]) / row["source_folder"]
        stem = source.name
        records.append(VolumeRecord(
            case_id=case_id,
            patient_id=str(row["patient_id"]),
            split=split,
            ct_path=source / "lung_ddpm_3d_128" / f"{stem}_ct_128.nii.gz",
            lung_path=source / "lung_ddpm_3d_128" / f"{stem}_lung_mask_128.nii.gz",
            roi_path=source / "lung_ddpm_3d_128" / f"{stem}_roi_128.nii.gz",
            labels=tuple(int(value) for value in row["roi_labels"] if int(value) > 0),
        ))
    return sorted(records, key=lambda item: item.case_id)


def require_prepared(records: list[VolumeRecord]) -> None:
    missing = [path for record in records for path in (record.ct_path, record.lung_path, record.roi_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} prepared 128^3 files; first missing: {missing[0]}. "
            "Run prepare_ild_3d_128.py first."
        )


class ILDVolumeDataset3D(Dataset):
    def __init__(self, records: list[VolumeRecord], augment: bool = False):
        self.records = records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        ct_image = nib.load(record.ct_path)
        ct = np.asarray(ct_image.dataobj, dtype=np.float32)
        lung = np.asarray(nib.load(record.lung_path).dataobj, dtype=np.uint8)
        roi = np.asarray(nib.load(record.roi_path).dataobj, dtype=np.uint8)
        # NIfTI x,y,z -> PyTorch depth,height,width.
        ct, lung, roi = (np.transpose(value, (2, 1, 0)).copy() for value in (ct, lung, roi))
        ct = np.clip(ct, -1000, 400)
        ct = (ct + 1000) / 700 - 1
        if self.augment:
            for axis in (1, 2):
                if random.random() < 0.5:
                    ct, lung, roi = (np.flip(value, axis=axis).copy() for value in (ct, lung, roi))
            if random.random() < 0.5:
                ct = np.clip(ct * random.uniform(0.95, 1.05) + random.uniform(-0.03, 0.03), -1, 1)
        ct_tensor = torch.from_numpy(ct).unsqueeze(0).float()
        lung_tensor = torch.from_numpy(lung).unsqueeze(0).float()
        roi_tensor = torch.from_numpy(roi.astype(np.int64))
        one_hot = F.one_hot(roi_tensor, num_classes=NUM_ROI_CLASSES + 1)[..., 1:]
        disease = one_hot.permute(3, 0, 1, 2).float()
        return {
            "image": ct_tensor,
            "condition": torch.cat((lung_tensor, disease), dim=0),
            "lung": lung_tensor,
            "roi": roi_tensor,
            "case_id": record.case_id,
            "patient_id": record.patient_id,
            "affine": torch.from_numpy(ct_image.affine.copy()),
        }


def label_balanced_sampler(records: list[VolumeRecord], seed: int, samples_per_epoch: int | None = None):
    counts = {label: sum(label in record.labels for record in records) for label in range(1, 18)}
    scores = []
    for record in records:
        available = [1 / counts[label] for label in record.labels if counts[label]]
        scores.append(max(available, default=1 / len(records)))
    weights = np.sqrt(np.asarray(scores, dtype=np.float64))
    weights /= weights.sum()
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=samples_per_epoch or len(records),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


class GaussianDiffusion3D(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, volume_size: int = 128, timesteps: int = 250):
        super().__init__()
        self.model, self.volume_size, self.num_timesteps = model, volume_size, timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        values = {
            "sqrt_alpha_bar": alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar": (1 - alpha_bar).sqrt(),
            "sqrt_recip_alpha_bar": (1 / alpha_bar).sqrt(),
            "sqrt_recipm1_alpha_bar": (1 / alpha_bar - 1).sqrt(),
            "posterior_variance": betas * (1 - alpha_bar_prev) / (1 - alpha_bar),
            "posterior_mean_coef1": betas * alpha_bar_prev.sqrt() / (1 - alpha_bar),
            "posterior_mean_coef2": (1 - alpha_bar_prev) * alphas.sqrt() / (1 - alpha_bar),
        }
        for name, value in values.items():
            self.register_buffer(name, value)

    def q_sample(self, clean, timesteps, noise=None):
        noise = torch.randn_like(clean) if noise is None else noise
        return extract(self.sqrt_alpha_bar, timesteps, clean.shape) * clean + extract(
            self.sqrt_one_minus_alpha_bar, timesteps, clean.shape
        ) * noise

    def forward(self, clean, condition, outside_weight=0.1, lung_weight=1.0,
                roi_weight=8.0, condition_dropout=0.1):
        timesteps = torch.randint(0, self.num_timesteps, (clean.shape[0],), device=clean.device)
        noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timesteps, noise)
        model_condition = condition.clone()
        if condition_dropout:
            drop = torch.rand((clean.shape[0], 1, 1, 1, 1), device=clean.device) < condition_dropout
            model_condition[:, 1:] = torch.where(drop, 0, model_condition[:, 1:])
        predicted = self.model(torch.cat((noisy, model_condition), dim=1), timesteps)
        weights = torch.full_like(clean, outside_weight)
        weights = torch.where(condition[:, :1] > 0.5, lung_weight, weights)
        weights = torch.where(condition[:, 1:].sum(1, keepdim=True) > 0.5, roi_weight, weights)
        return (torch.abs(predicted - noise) * weights).sum() / weights.sum().clamp_min(1)

    @torch.no_grad()
    def sample(self, condition, reference, lung, strength=1.0, guidance_scale=1.5):
        reference_noise = torch.randn_like(reference)
        start_step = int(round((self.num_timesteps - 1) * strength))
        if strength < 1:
            start_t = torch.full((condition.shape[0],), start_step, device=condition.device, dtype=torch.long)
            image = self.q_sample(reference, start_t, reference_noise)
        else:
            image = torch.randn_like(reference)
        for step in reversed(range(start_step + 1)):
            t = torch.full((condition.shape[0],), step, device=image.device, dtype=torch.long)
            conditional = self.model(torch.cat((image, condition), 1), t)
            if guidance_scale == 1:
                predicted_noise = conditional
            else:
                null = condition.clone(); null[:, 1:] = 0
                unconditional = self.model(torch.cat((image, null), 1), t)
                predicted_noise = unconditional + guidance_scale * (conditional - unconditional)
            clean = (extract(self.sqrt_recip_alpha_bar, t, image.shape) * image -
                     extract(self.sqrt_recipm1_alpha_bar, t, image.shape) * predicted_noise).clamp(-1, 1)
            mean = extract(self.posterior_mean_coef1, t, image.shape) * clean + extract(
                self.posterior_mean_coef2, t, image.shape
            ) * image
            image = mean if step == 0 else mean + extract(
                self.posterior_variance, t, image.shape
            ).clamp_min(1e-20).sqrt() * torch.randn_like(image)
            if step:
                previous_t = torch.full((condition.shape[0],), step - 1, device=image.device, dtype=torch.long)
                outside_reference = self.q_sample(reference, previous_t, reference_noise)
            else:
                outside_reference = reference
            image = torch.where(lung > 0.5, image, outside_reference)
        return image.clamp(-1, 1)


__all__ = [
    "NUM_CONDITION_CHANNELS_3D", "VolumeRecord", "records_from_nnunet_report",
    "require_prepared", "ILDVolumeDataset3D", "label_balanced_sampler",
    "GaussianDiffusion3D", "update_ema", "seed_everything",
]
