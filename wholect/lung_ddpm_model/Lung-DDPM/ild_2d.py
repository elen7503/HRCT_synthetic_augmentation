"""Utilities for 2-D semantic-layout-conditioned ILD CT synthesis."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


LABEL_NAMES = {
    1: "healthy_control",
    2: "emphysema",
    3: "ground_glass",
    4: "fibrosis",
    5: "micronodules",
    6: "consolidation",
}
NUM_DISEASE_CLASSES = len(LABEL_NAMES)
NUM_CONDITION_CHANNELS = 1 + NUM_DISEASE_CLASSES  # lung + six ILD classes


@dataclass(frozen=True)
class SliceRecord:
    patient_dir: Path
    slice_id: str
    patient_id: str
    labels: tuple[int, ...]
    roi_file: Path | None

    @property
    def image_path(self) -> Path:
        return self.patient_dir / "images" / f"{self.slice_id}.npy"

    @property
    def lung_path(self) -> Path:
        return self.patient_dir / "lung_masks" / f"{self.slice_id}.npy"

    @property
    def roi_path(self) -> Path | None:
        return self.roi_file

    @property
    def has_disease_condition(self) -> bool:
        """Whether this slice has an explicit class-labelled ROI.

        A slice without an ROI is an *unknown/null-condition* sample. It must
        not be interpreted as a confirmed healthy or background-only slice.
        """
        return bool(self.labels)


def patient_id(folder_name: str) -> str:
    name = folder_name.removeprefix("patient_")
    match = re.match(r"^(\d+)", name)
    return match.group(1) if match else name


def split_patient_ids(report_path: Path, split: str) -> set[str]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        str(item["patient_id"])
        for item in report["case_mapping"].values()
        if item["split"] == split
    }


def training_patient_ids(report_path: Path) -> set[str]:
    return split_patient_ids(report_path, "train")


def discover_records(
    data_root: Path,
    allowed_patients: set[str] | None = None,
    require_foreground: bool = True,
    max_label: int = NUM_DISEASE_CLASSES,
) -> list[SliceRecord]:
    """Discover aligned CT/lung/optional-ROI slices.

    Discovery is image-led so that CT + lung-mask slices remain available even
    when no ROI file exists. Empty or absent ROIs become null-condition samples;
    only explicit labels in ``1..max_label`` become disease conditions.
    """
    records: list[SliceRecord] = []
    for folder in sorted(path for path in data_root.glob("patient_*") if path.is_dir()):
        pid = patient_id(folder.name)
        if allowed_patients is not None and pid not in allowed_patients:
            continue
        for image_path in sorted((folder / "images").glob("slice_*.npy")):
            slice_id = image_path.stem
            lung_path = folder / "lung_masks" / f"{slice_id}.npy"
            if not lung_path.exists():
                raise FileNotFoundError(f"Unaligned ILD slice: {folder.name}/{slice_id}")
            candidate = folder / "roi_masks" / f"{slice_id}.npy"
            roi_path = candidate if candidate.exists() else None
            if roi_path is None:
                labels: tuple[int, ...] = ()
            else:
                roi = np.load(roi_path, mmap_mode="r")
                labels = tuple(
                    int(value)
                    for value in np.unique(roi)
                    if 1 <= int(value) <= max_label
                )
            if require_foreground and not labels:
                continue
            records.append(SliceRecord(folder, slice_id, pid, labels, roi_path))
    if not records:
        raise ValueError(f"No eligible slices found in {data_root}")
    return records


def resize_2d(array: np.ndarray, size: int, nearest: bool) -> np.ndarray:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = Image.fromarray(array)
    return np.asarray(image.resize((size, size), resample=mode))


def make_condition(lung: np.ndarray, roi: np.ndarray) -> np.ndarray:
    condition = np.zeros((NUM_CONDITION_CHANNELS, *roi.shape), dtype=np.float32)
    condition[0] = lung > 0
    for label in LABEL_NAMES:
        condition[label] = roi == label
    return condition


def center_crop(array: np.ndarray, center_y: int, center_x: int, crop_size: int) -> np.ndarray:
    height, width = array.shape
    crop_size = min(crop_size, height, width)
    y0 = min(max(center_y - crop_size // 2, 0), height - crop_size)
    x0 = min(max(center_x - crop_size // 2, 0), width - crop_size)
    return array[y0 : y0 + crop_size, x0 : x0 + crop_size]


def affine_2d(
    array: np.ndarray,
    angle: float,
    scale: float,
    nearest: bool,
    fill: int,
) -> np.ndarray:
    image = Image.fromarray(array)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = image.rotate(angle, resample=resample, fillcolor=fill)
    width, height = image.size
    inverse_scale = 1.0 / scale
    cx, cy = width / 2.0, height / 2.0
    matrix = (
        inverse_scale,
        0.0,
        cx - inverse_scale * cx,
        0.0,
        inverse_scale,
        cy - inverse_scale * cy,
    )
    return np.asarray(
        image.transform(image.size, Image.Transform.AFFINE, matrix, resample=resample, fillcolor=fill)
    )


class ILDLayoutDataset2D(Dataset):
    def __init__(
        self,
        records: list[SliceRecord],
        image_size: int = 256,
        augment: bool = False,
        roi_crop_probability: float = 0.0,
        roi_crop_size: int = 384,
        rotation_degrees: float = 10.0,
        scale_range: tuple[float, float] = (0.9, 1.1),
        intensity_jitter: float = 0.05,
    ):
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.roi_crop_probability = roi_crop_probability
        self.roi_crop_size = roi_crop_size
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.intensity_jitter = intensity_jitter
        if not 0 <= roi_crop_probability <= 1:
            raise ValueError("roi_crop_probability must be between 0 and 1")
        if roi_crop_size <= 0:
            raise ValueError("roi_crop_size must be positive")
        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError("scale_range must be positive and ordered")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        image = np.load(record.image_path).astype(np.uint8)
        lung = (np.load(record.lung_path) > 0).astype(np.uint8)
        roi = (
            np.load(record.roi_path).astype(np.uint8)
            if record.roi_path is not None
            else np.zeros_like(image, dtype=np.uint8)
        )
        if image.shape != lung.shape or image.shape != roi.shape:
            raise ValueError(f"Geometry mismatch: {record.patient_dir.name}/{record.slice_id}")

        target_roi = (roi >= 1) & (roi <= NUM_DISEASE_CLASSES)
        if self.augment and random.random() < self.roi_crop_probability and np.any(target_roi):
            coordinates = np.argwhere(target_roi)
            center_y, center_x = coordinates[random.randrange(len(coordinates))]
            image = center_crop(image, int(center_y), int(center_x), self.roi_crop_size)
            lung = center_crop(lung, int(center_y), int(center_x), self.roi_crop_size)
            roi = center_crop(roi, int(center_y), int(center_x), self.roi_crop_size)

        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                lung = np.fliplr(lung).copy()
                roi = np.fliplr(roi).copy()
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            scale = random.uniform(*self.scale_range)
            image_fill = int(np.median(np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))))
            image = affine_2d(image, angle, scale, nearest=False, fill=image_fill)
            lung = affine_2d(lung, angle, scale, nearest=True, fill=0)
            roi = affine_2d(roi, angle, scale, nearest=True, fill=0)
            if self.intensity_jitter > 0:
                gain = random.uniform(1.0 - self.intensity_jitter, 1.0 + self.intensity_jitter)
                shift = random.uniform(-255 * self.intensity_jitter, 255 * self.intensity_jitter)
                image = np.clip(image.astype(np.float32) * gain + shift, 0, 255).astype(np.uint8)

        image = resize_2d(image, self.image_size, nearest=False)
        lung = resize_2d(lung, self.image_size, nearest=True)
        roi = resize_2d(roi, self.image_size, nearest=True)
        condition = make_condition(lung, roi)
        target = image.astype(np.float32)[None] / 127.5 - 1.0
        return {
            "image": torch.from_numpy(target),
            "condition": torch.from_numpy(condition),
            "lung": torch.from_numpy(lung[None].astype(np.float32)),
            "roi": torch.from_numpy(roi.astype(np.int64)),
            "has_disease_condition": torch.as_tensor(record.has_disease_condition),
            "patient_id": record.patient_id,
            "slice_id": record.slice_id,
        }


def class_counts(records: list[SliceRecord]) -> dict[int, int]:
    return {
        label: sum(label in record.labels for record in records)
        for label in LABEL_NAMES
    }


def dataset_manifest(records: list[SliceRecord], data_root: Path) -> dict:
    """Create a reproducibility manifest for the exact arrays used by training."""
    rows = []
    digest = hashlib.sha256()
    for record in records:
        files = {}
        source_files = [("image", record.image_path), ("lung", record.lung_path)]
        if record.roi_path is not None:
            source_files.append(("roi", record.roi_path))
        for name, path in source_files:
            file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            files[name] = {
                "path": str(path.resolve().relative_to(data_root.resolve())),
                "sha256": file_hash,
            }
            digest.update(name.encode("utf-8"))
            digest.update(file_hash.encode("ascii"))
        rows.append(
            {
                "patient_id": record.patient_id,
                "slice_id": record.slice_id,
                "labels": list(record.labels),
                "condition_kind": (
                    "disease_roi" if record.has_disease_condition else "null_unknown"
                ),
                "files": files,
            }
        )
    return {
        "schema_version": 1,
        "num_records": len(rows),
        "sha256": digest.hexdigest(),
        "records": rows,
    }


def mixed_condition_sampler(
    records: list[SliceRecord],
    seed: int,
    samples_per_epoch: int | None = None,
    roi_sample_fraction: float = 0.5,
) -> WeightedRandomSampler:
    """Balance null-condition anatomy samples and class-labelled ROI samples.

    ``roi_sample_fraction`` controls the expected mass assigned to ROI-positive
    slices. Within that group, rare disease classes are oversampled. The null
    group is sampled uniformly and is never treated as confirmed healthy.
    """
    if not 0 <= roi_sample_fraction <= 1:
        raise ValueError("roi_sample_fraction must be between 0 and 1")
    roi_indices = [i for i, record in enumerate(records) if record.labels]
    null_indices = [i for i, record in enumerate(records) if not record.labels]
    if not roi_indices and roi_sample_fraction > 0:
        raise ValueError("roi_sample_fraction is positive but no disease ROI records exist")
    if not null_indices and roi_sample_fraction < 1:
        roi_sample_fraction = 1.0

    counts = class_counts(records)
    weights = np.zeros(len(records), dtype=np.float64)
    if roi_indices:
        roi_scores = np.asarray(
            [max(1.0 / counts[label] for label in records[i].labels) for i in roi_indices],
            dtype=np.float64,
        )
        roi_scores /= roi_scores.sum()
        weights[roi_indices] = roi_scores * roi_sample_fraction
    if null_indices:
        weights[null_indices] = (1.0 - roi_sample_fraction) / len(null_indices)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=samples_per_epoch or len(records),
        replacement=True,
        generator=generator,
    )


# Backwards-compatible name for external callers. New training code should use
# ``mixed_condition_sampler`` explicitly so the sampling semantics are visible.
balanced_sampler = mixed_condition_sampler


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0, 0.999).float()


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values.gather(0, timesteps).reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion2D(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, image_size: int, timesteps: int = 250):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.num_timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
        for name, value in {
            "betas": betas,
            "alphas": alphas,
            "alpha_bar": alpha_bar,
            "alpha_bar_prev": alpha_bar_prev,
            "sqrt_alpha_bar": alpha_bar.sqrt(),
            "sqrt_one_minus_alpha_bar": (1.0 - alpha_bar).sqrt(),
            "sqrt_recip_alpha_bar": (1.0 / alpha_bar).sqrt(),
            "sqrt_recipm1_alpha_bar": (1.0 / alpha_bar - 1).sqrt(),
            "posterior_variance": betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar),
            "posterior_mean_coef1": betas * alpha_bar_prev.sqrt() / (1.0 - alpha_bar),
            "posterior_mean_coef2": (1.0 - alpha_bar_prev) * alphas.sqrt() / (1.0 - alpha_bar),
        }.items():
            self.register_buffer(name, value)

    def q_sample(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        noise = torch.randn_like(clean) if noise is None else noise
        return (
            extract(self.sqrt_alpha_bar, timesteps, clean.shape) * clean
            + extract(self.sqrt_one_minus_alpha_bar, timesteps, clean.shape) * noise
        )

    def forward(
        self,
        clean: torch.Tensor,
        condition: torch.Tensor,
        outside_weight: float = 0.25,
        lung_weight: float = 1.0,
        roi_weight: float = 5.0,
        class_weights: tuple[float, ...] = (1.0, 2.0, 2.0, 1.0, 2.0, 3.0),
        condition_dropout: float = 0.1,
        max_training_strength: float = 1.0,
    ) -> torch.Tensor:
        if len(class_weights) != NUM_DISEASE_CLASSES:
            raise ValueError(f"class_weights must contain {NUM_DISEASE_CLASSES} values")
        if not 0 <= condition_dropout <= 1:
            raise ValueError("condition_dropout must be between 0 and 1")
        if not 0 < max_training_strength <= 1:
            raise ValueError("max_training_strength must be in (0, 1]")
        max_timestep = max(1, int(round((self.num_timesteps - 1) * max_training_strength)) + 1)
        timesteps = torch.randint(
            0, max_timestep, (clean.shape[0],), device=clean.device
        )
        noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timesteps, noise)
        model_condition = condition
        if condition_dropout > 0:
            drop = (
                torch.rand((clean.shape[0], 1, 1, 1), device=clean.device)
                < condition_dropout
            )
            model_condition = condition.clone()
            model_condition[:, 1:] = torch.where(
                drop, torch.zeros_like(model_condition[:, 1:]), model_condition[:, 1:]
            )
        predicted = self.model(torch.cat((noisy, model_condition), dim=1), timesteps)

        weights = torch.full_like(clean, outside_weight)
        lung = condition[:, :1] > 0.5
        weights = torch.where(lung, torch.as_tensor(lung_weight, device=clean.device), weights)
        for label_index, class_weight in enumerate(class_weights, start=1):
            class_mask = condition[:, label_index : label_index + 1] > 0.5
            label_weight = roi_weight * class_weight
            weights = torch.where(
                class_mask, torch.as_tensor(label_weight, device=clean.device), weights
            )
        pixel_loss = torch.abs(predicted - noise)
        return (pixel_loss * weights).sum() / weights.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        reference: torch.Tensor | None = None,
        lung: torch.Tensor | None = None,
        guidance_scale: float = 1.0,
        blend_mode: str = "lung",
        inpainting_strength: float = 1.0,
    ) -> torch.Tensor:
        if guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")
        if blend_mode not in ("lung", "roi", "none"):
            raise ValueError("blend_mode must be one of: lung, roi, none")
        if not 0 < inpainting_strength <= 1:
            raise ValueError("inpainting_strength must be in (0, 1]")
        # Use one coherent reference-noise realization throughout the reverse
        # trajectory. Redrawing it at every timestep creates temporally
        # inconsistent context at the inpainting boundary.
        reference_noise = torch.randn_like(reference) if reference is not None else None
        if reference is not None:
            # SDEdit-style image-to-image initialization. A strength below 1
            # preserves patient anatomy and asks the model to modify texture,
            # rather than synthesizing all ROI structure from pure noise.
            start_step = int(round((self.num_timesteps - 1) * inpainting_strength))
            start_t = torch.full(
                (condition.shape[0],), start_step, device=condition.device, dtype=torch.long
            )
            image = self.q_sample(reference, start_t, reference_noise)
        else:
            start_step = self.num_timesteps - 1
            image = torch.randn(
                (condition.shape[0], 1, self.image_size, self.image_size),
                device=condition.device,
            )
        for step in reversed(range(start_step + 1)):
            t = torch.full((condition.shape[0],), step, device=image.device, dtype=torch.long)
            predicted_conditional = self.model(torch.cat((image, condition), dim=1), t)
            if guidance_scale == 1:
                predicted_noise = predicted_conditional
            else:
                unconditional = condition.clone()
                unconditional[:, 1:] = 0
                predicted_unconditional = self.model(
                    torch.cat((image, unconditional), dim=1), t
                )
                predicted_noise = predicted_unconditional + guidance_scale * (
                    predicted_conditional - predicted_unconditional
                )
            predicted_clean = (
                extract(self.sqrt_recip_alpha_bar, t, image.shape) * image
                - extract(self.sqrt_recipm1_alpha_bar, t, image.shape) * predicted_noise
            ).clamp(-1, 1)
            mean = (
                extract(self.posterior_mean_coef1, t, image.shape) * predicted_clean
                + extract(self.posterior_mean_coef2, t, image.shape) * image
            )
            if step:
                variance = extract(self.posterior_variance, t, image.shape).clamp_min(1e-20)
                image = mean + variance.sqrt() * torch.randn_like(image)
            else:
                image = mean
            if reference is not None and blend_mode != "none":
                if blend_mode == "lung":
                    if lung is None:
                        raise ValueError("lung is required when blend_mode='lung'")
                    generation_mask = lung > 0.5
                else:
                    generation_mask = condition[:, 1:].sum(1, keepdim=True) > 0.5
                outside = ~generation_mask
                if step:
                    # `image` is x_(t-1) after the reverse update, therefore
                    # the known reference region must also be noised at t-1.
                    reference_t = torch.full(
                        (condition.shape[0],), step - 1,
                        device=image.device, dtype=torch.long,
                    )
                    reference_value = self.q_sample(reference, reference_t, reference_noise)
                else:
                    reference_value = reference
                image = torch.where(outside, reference_value, image)
        return image.clamp(-1, 1)


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.data.mul_(decay).add_(parameter.data, alpha=1.0 - decay)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
