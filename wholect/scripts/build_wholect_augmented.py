"""
Builds two datasets for a fair baseline-vs-augmented comparison, both
using the SAME lung-window uint8 preprocessing
"""

import os
import json
import glob

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import cv2

REAL_WHOLECT_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_npy")
GENERATED_DIR = os.path.join(PROJECT_ROOT, "wholect/lung_ddpm_model/Lung-DDPM/generated/ILD-DDPM-2D-V3-balanced-test")

OUT_REAL_ONLY_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_uint8")
OUT_AUGMENTED_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_augmented")

LUNGDDPM_LABEL_TO_YOUR_CLASS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]

UNMATCHED_SOURCE_SENTINEL = -100


def apply_lung_window(hu_image, center=-600, width=1600):
    lo = center - width / 2
    hi = center + width / 2
    clipped = np.clip(hu_image, lo, hi).astype(np.float32)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def main():
    os.makedirs(OUT_REAL_ONLY_DIR, exist_ok=True)
    os.makedirs(OUT_AUGMENTED_DIR, exist_ok=True)

    real_images_hu = np.load(os.path.join(REAL_WHOLECT_DIR, "all_images.npy"))
    real_labels = np.load(os.path.join(REAL_WHOLECT_DIR, "all_labels.npy"))
    real_pids = np.load(os.path.join(REAL_WHOLECT_DIR, "all_patient_ids.npy"))

    real_images_uint8 = apply_lung_window(real_images_hu)

    print(f"Real: {len(real_images_uint8)} slices, {len(np.unique(real_pids))} patients")

    np.save(os.path.join(OUT_REAL_ONLY_DIR, "all_images.npy"), real_images_uint8)
    np.save(os.path.join(OUT_REAL_ONLY_DIR, "all_labels.npy"), real_labels)
    np.save(os.path.join(OUT_REAL_ONLY_DIR, "all_patient_ids.npy"), real_pids)
    print(f"Saved real-only (lung-window uint8) dataset to {OUT_REAL_ONLY_DIR}/")

    report_path = os.path.join(GENERATED_DIR, "generation_report.json")
    with open(report_path) as f:
        report = json.load(f)

    synth_images, synth_labels, synth_pids = [], [], []
    n_unmatched = 0

    for case_info in report["cases"]:
        lungddpm_label = case_info["target_class"]
        if lungddpm_label not in LUNGDDPM_LABEL_TO_YOUR_CLASS:
            continue
        your_class = LUNGDDPM_LABEL_TO_YOUR_CLASS[lungddpm_label]

        source_patient_str = case_info["source_patient"]
        try:
            source_pid = int(source_patient_str)
        except ValueError:
            source_pid = UNMATCHED_SOURCE_SENTINEL
            n_unmatched += 1

        case_dir = os.path.join(GENERATED_DIR, case_info["case"], "images")
        for f in glob.glob(os.path.join(case_dir, "*.npy")):
            arr = np.load(f)
            if arr.shape != (512, 512):
                arr = cv2.resize(arr.astype(np.float32), (512, 512),
                                 interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            synth_images.append(arr)
            synth_labels.append(your_class)
            synth_pids.append(source_pid)

    synth_images = np.stack(synth_images)
    synth_labels = np.array(synth_labels, dtype=np.int64)
    synth_pids = np.array(synth_pids, dtype=np.int64)

    print(f"\nSynthetic: {len(synth_images)} slices")
    print(f"  Tagged with real numeric source patient ID: {len(synth_images) - n_unmatched}")
    print(f"  Tagged with sentinel (non-numeric source): {n_unmatched}")

    all_images = np.concatenate([real_images_uint8, synth_images], axis=0)
    all_labels = np.concatenate([real_labels, synth_labels], axis=0)
    all_pids = np.concatenate([real_pids, synth_pids], axis=0)

    np.save(os.path.join(OUT_AUGMENTED_DIR, "all_images.npy"), all_images)
    np.save(os.path.join(OUT_AUGMENTED_DIR, "all_labels.npy"), all_labels)
    np.save(os.path.join(OUT_AUGMENTED_DIR, "all_patient_ids.npy"), all_pids)

    print(f"\nSaved augmented dataset to {OUT_AUGMENTED_DIR}/")
    print(f"  Total: {len(all_images)}  (real: {len(real_images_uint8)}, synthetic: {len(synth_images)})")
    print("\nPer-class totals:")
    for c, name in enumerate(CLASS_NAMES):
        n_real = int((real_labels == c).sum())
        n_synth = int((synth_labels == c).sum())
        print(f"  {name:15s} real={n_real:4d}  synthetic={n_synth:4d}  total={n_real+n_synth:4d}")


if __name__ == "__main__":
    main()
