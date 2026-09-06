"""
build_wholect_dataset.py
----------------------------
Builds a whole-slice-level classification dataset from ILD_DB_volumeROIs:
for every CT slice that has at least one of your 5 target classes
annotated, assigns that slice a single dominant-class label (by pixel
count) and saves it alongside the matching CT slice image.

CT intensities are converted to true HU via apply_modality_lut (reads
RescaleSlope/RescaleIntercept per-file) -- raw pixel_array values are
NOT Hounsfield Units on their own.

Output (mirrors your patch dataset's structure):
  all_images.npy      -- (N, 512, 512) int16, true HU CT slices
  all_labels.npy      -- (N,) int64, dominant target class (0-4)
  all_patient_ids.npy -- (N,) int64, patient ID for LOPO grouping

Usage:
  python build_wholect_dataset.py
"""

import os
import re
import glob

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

VOLUMEROIS_DIR = os.path.join(PROJECT_ROOT, "ILD_DB/ILD_DB_volumeROIs")
OUT_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_npy")

TARGET_LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]

MIN_PIXELS = 20


def extract_slice_num(filename):
    m = re.search(r'_(\d+)\.dcm$', filename)
    return int(m.group(1)) if m else None


def find_matching_ct(patient_dir, slice_num):
    candidates = glob.glob(os.path.join(patient_dir, f"CT-*-{slice_num:04d}.dcm"))
    if not candidates:
        candidates = [f for f in glob.glob(os.path.join(patient_dir, "CT-*.dcm"))
                     if extract_slice_num(os.path.basename(f).replace("CT-", "roi_mask_")) == slice_num]
    return candidates[0] if candidates else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    patient_dirs = sorted(glob.glob(os.path.join(VOLUMEROIS_DIR, "*")))

    all_images, all_labels, all_pids = [], [], []
    skipped_no_ct_match = 0
    skipped_no_target_class = 0
    per_class_counts = {name: 0 for name in CLASS_NAMES}

    for pdir in patient_dirs:
        pid_str = os.path.basename(pdir)
        try:
            pid = int(pid_str)
        except ValueError:
            continue

        roi_dir = os.path.join(pdir, "roi_mask")
        if not os.path.isdir(roi_dir):
            continue

        for mask_path in glob.glob(os.path.join(roi_dir, "*.dcm")):
            slice_num = extract_slice_num(os.path.basename(mask_path))
            if slice_num is None:
                continue

            try:
                mask_arr = pydicom.dcmread(mask_path).pixel_array
            except Exception:
                continue

            if mask_arr.shape != (512, 512):
                mask_arr = cv2.resize(mask_arr.astype(np.float32), (512, 512),
                                      interpolation=cv2.INTER_NEAREST).astype(mask_arr.dtype)

            vals, counts = np.unique(mask_arr, return_counts=True)
            target_counts = {}
            for v, c in zip(vals, counts):
                v = int(v)
                if v in TARGET_LABEL_MAP and c >= MIN_PIXELS:
                    target_counts[TARGET_LABEL_MAP[v]] = int(c)

            if not target_counts:
                skipped_no_target_class += 1
                continue

            dominant_class = max(target_counts, key=target_counts.get)

            ct_path = find_matching_ct(pdir, slice_num)
            if ct_path is None:
                skipped_no_ct_match += 1
                continue

            try:
                ct_ds = pydicom.dcmread(ct_path)
                ct_arr = apply_modality_lut(ct_ds.pixel_array, ct_ds).astype(np.int16)
            except Exception:
                skipped_no_ct_match += 1
                continue

            if ct_arr.shape != (512, 512):
                ct_arr = cv2.resize(ct_arr.astype(np.float32), (512, 512),
                                     interpolation=cv2.INTER_LINEAR).astype(np.int16)

            all_images.append(ct_arr)
            all_labels.append(dominant_class)
            all_pids.append(pid)
            per_class_counts[CLASS_NAMES[dominant_class]] += 1

    if not all_images:
        print("ERROR: no slices collected. Check VOLUMEROIS_DIR and file naming patterns.")
        return

    all_images = np.stack(all_images, axis=0)
    all_labels = np.array(all_labels, dtype=np.int64)
    all_pids = np.array(all_pids, dtype=np.int64)

    np.save(os.path.join(OUT_DIR, "all_images.npy"), all_images)
    np.save(os.path.join(OUT_DIR, "all_labels.npy"), all_labels)
    np.save(os.path.join(OUT_DIR, "all_patient_ids.npy"), all_pids)

    print(f"Saved {len(all_images)} whole-CT slices to {OUT_DIR}/")
    print(f"  Image shape: {all_images.shape[1:]}, dtype: {all_images.dtype}")
    print(f"  HU range: [{all_images.min()}, {all_images.max()}]")
    print(f"  Distinct patients: {len(np.unique(all_pids))}")
    print(f"  Skipped (no target class in slice): {skipped_no_target_class}")
    print(f"  Skipped (no matching CT slice found): {skipped_no_ct_match}")
    print("\nPer-class slice counts:")
    for name, count in per_class_counts.items():
        n_patients = len(np.unique(all_pids[all_labels == CLASS_NAMES.index(name)]))
        print(f"  {name:15s}: {count:5d} slices  ({n_patients} distinct patients)")


if __name__ == "__main__":
    main()
