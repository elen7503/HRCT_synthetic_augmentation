#!/usr/bin/env python3
"""Prepare lung-centered 128^3 CT/lung/ROI volumes for 3-D Lung-DDPM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def mask_center(mask: sitk.Image) -> np.ndarray:
    coords = np.argwhere(sitk.GetArrayViewFromImage(mask) > 0)
    center_zyx = coords.mean(0) if len(coords) else (np.asarray(mask.GetSize()) - 1)[::-1] / 2
    return np.asarray(mask.TransformContinuousIndexToPhysicalPoint(tuple(center_zyx[::-1])))


def geometry(ct: sitk.Image, lung: sitk.Image, size: int, z_spacing: float):
    old_size, old_spacing = np.asarray(ct.GetSize()), np.asarray(ct.GetSpacing())
    spacing = np.asarray([old_size[0] * old_spacing[0] / size,
                          old_size[1] * old_spacing[1] / size, z_spacing])
    direction = np.asarray(ct.GetDirection()).reshape(3, 3)
    origin = mask_center(lung) - direction @ (((np.full(3, size) - 1) / 2) * spacing)
    return tuple(spacing), tuple(origin), tuple(ct.GetDirection())


def convert(patient: Path, size: int, z_spacing: float, overwrite: bool) -> dict:
    stem, output = patient.name, patient / "lung_ddpm_3d_128"
    output.mkdir(exist_ok=True)
    sources = [patient / f"{stem}_ct.nii.gz", patient / f"{stem}_lung_mask.nii.gz",
               patient / f"{stem}_roi.nii.gz"]
    targets = [output / f"{stem}_ct_128.nii.gz", output / f"{stem}_lung_mask_128.nii.gz",
               output / f"{stem}_roi_128.nii.gz"]
    if not overwrite and all(path.exists() for path in targets):
        return {"patient": stem, "status": "skipped"}
    ct, lung, roi = (sitk.ReadImage(str(path)) for path in sources)
    spacing, origin, direction = geometry(ct, lung, size, z_spacing)
    settings = [
        (ct, sitk.sitkLinear, -1000, sitk.sitkInt16),
        (lung, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8),
        (roi, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8),
    ]
    for (image, interpolation, default, pixel_type), target in zip(settings, targets):
        converted = sitk.Resample(image, [size] * 3, sitk.Transform(), interpolation,
                                  origin, spacing, direction, default, pixel_type)
        sitk.WriteImage(converted, str(target), True)
    return {"patient": stem, "status": "written", "spacing": list(spacing)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--z-spacing", type=float, default=2.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    patients = sorted(path for path in args.data_root.resolve().glob("patient_*") if path.is_dir())
    report = []
    for index, patient in enumerate(patients, 1):
        result = convert(patient, args.size, args.z_spacing, args.overwrite)
        report.append(result)
        print(f"[{index:03d}/{len(patients):03d}] {patient.name}: {result['status']}", flush=True)
    path = args.data_root / "lung_ddpm_3d_128_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
