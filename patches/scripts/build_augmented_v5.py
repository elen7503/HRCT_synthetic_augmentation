"""
build_augmented_v5.py
------------------------
Same structure as build_augmented_v3.py / v3_5k, but sources synthetic
patches from the QUALITY-FILTERED set (filter_by_quality_v5.py output)
instead of raw unfiltered sampling. Dose is fixed at 5000/class, matching
v3_5k, so this comparison isolates the effect of quality filtering alone.

Usage:
  python build_augmented_v5.py
"""

import os

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np

REAL_DIR = os.path.join(PROJECT_ROOT, "ILD_DB_npy")
SYNTH_DIR = os.path.join(PROJECT_ROOT, "diffusion_model/outputs/synthetic_v5")
OUT_DIR = os.path.join(PROJECT_ROOT, "ILD_DB_npy_augmented_v5")

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

    print(f"Real data (kept in full): {len(real_images)} patches\n")
    EXCLUDED_CLASSES = {"emphysema", "ground_glass"}

    for class_idx, class_name in enumerate(CLASS_NAMES):
        if class_name in EXCLUDED_CLASSES:
            print(f"  [EXCLUDED] {class_name}: no synthetic added (filter found no "
                  f"acceptable-confidence samples in the pool) -- real data only")
            continue

        img_path = os.path.join(SYNTH_DIR, f"synthetic_{class_name}_images.npy")
        lbl_path = os.path.join(SYNTH_DIR, f"synthetic_{class_name}_labels.npy")

        if not os.path.exists(img_path):
            print(f"  [MISSING] {class_name}: no filtered synthetic file found")
            continue

        synth_images = np.load(img_path)
        synth_labels = np.load(lbl_path)
        synth_pids = np.full(len(synth_images), SYNTHETIC_PATIENT_ID, dtype=np.int64)

        all_images.append(synth_images)
        all_labels.append(synth_labels)
        all_pids.append(synth_pids)

        real_n = int((real_labels == class_idx).sum())
        print(f"  {class_name:15s} real={real_n:5d}  +synthetic(filtered)={len(synth_images):5d}  "
              f"total={real_n + len(synth_images):5d}")

    merged_images = np.concatenate(all_images, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    merged_pids = np.concatenate(all_pids, axis=0)

    assert len(merged_images) == len(merged_labels) == len(merged_pids)

    np.save(os.path.join(OUT_DIR, "all_images.npy"), merged_images)
    np.save(os.path.join(OUT_DIR, "all_labels.npy"), merged_labels)
    np.save(os.path.join(OUT_DIR, "all_patient_ids.npy"), merged_pids)

    print(f"\nSaved merged v5 dataset to {OUT_DIR}/")
    print(f"  Total: {len(merged_images)}  (real: {(merged_pids != SYNTHETIC_PATIENT_ID).sum()}, "
          f"synthetic: {(merged_pids == SYNTHETIC_PATIENT_ID).sum()})")


if __name__ == "__main__":
    main()