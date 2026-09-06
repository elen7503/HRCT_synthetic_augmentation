"""
build_augmented_v2.py
------------------------
Merges real patches (ILD_DB_npy) with the newly resampled synthetic
patches (outputs/synthetic_v2, from quality-selected checkpoints) into
a new augmented dataset: ILD_DB_npy_augmented_v2/

Synthetic patches are flagged patient_id = -2, same convention as before.

Usage:
  python build_augmented_v2.py
"""

import os

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np

REAL_DIR = os.path.join(PROJECT_ROOT, "ILD_DB_npy")
SYNTH_DIR = os.path.join(PROJECT_ROOT, "diffusion_model/outputs/synthetic_v2")
OUT_DIR = os.path.join(PROJECT_ROOT, "ILD_DB_npy_augmented_v2")

CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]
SYNTHETIC_PATIENT_ID = -2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    real_images = np.load(os.path.join(REAL_DIR, "all_images.npy"))
    real_labels = np.load(os.path.join(REAL_DIR, "all_labels.npy"))
    real_pids = np.load(os.path.join(REAL_DIR, "all_patient_ids.npy"))

    all_images = [real_images]
    all_labels = [real_labels]
    all_pids = [real_pids]

    print(f"Real data: {len(real_images)} patches")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        img_path = os.path.join(SYNTH_DIR, f"synthetic_{class_name}_images.npy")
        lbl_path = os.path.join(SYNTH_DIR, f"synthetic_{class_name}_labels.npy")
        if not os.path.exists(img_path):
            print(f"  {class_name}: no synthetic file found, skipping (expected for micronodules)")
            continue

        synth_images = np.load(img_path)
        synth_labels = np.load(lbl_path)
        synth_pids = np.full(len(synth_images), SYNTHETIC_PATIENT_ID, dtype=np.int64)

        all_images.append(synth_images)
        all_labels.append(synth_labels)
        all_pids.append(synth_pids)
        print(f"  {class_name}: +{len(synth_images)} synthetic patches (from quality-selected checkpoint)")

    merged_images = np.concatenate(all_images, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    merged_pids = np.concatenate(all_pids, axis=0)

    assert len(merged_images) == len(merged_labels) == len(merged_pids), "Length mismatch after merge!"

    np.save(os.path.join(OUT_DIR, "all_images.npy"), merged_images)
    np.save(os.path.join(OUT_DIR, "all_labels.npy"), merged_labels)
    np.save(os.path.join(OUT_DIR, "all_patient_ids.npy"), merged_pids)

    print(f"\nSaved merged dataset to {OUT_DIR}/")
    print(f"  Total: {len(merged_images)}  (real: {(merged_pids != SYNTHETIC_PATIENT_ID).sum()}, "
          f"synthetic: {(merged_pids == SYNTHETIC_PATIENT_ID).sum()})")

    print("\nPer-class breakdown:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = merged_labels == class_idx
        real_n = int((mask & (merged_pids != SYNTHETIC_PATIENT_ID)).sum())
        synth_n = int((mask & (merged_pids == SYNTHETIC_PATIENT_ID)).sum())
        print(f"  {class_name:15s} real={real_n:5d}  synthetic={synth_n:5d}  total={real_n+synth_n:5d}")


if __name__ == "__main__":
    main()