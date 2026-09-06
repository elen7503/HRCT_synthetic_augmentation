"""
build_augmented_v3_5k.py
------------------------
Builds the v3 augmented dataset: ALL real patches (no trimming) for every
class, plus a FIXED number of synthetic patches per class (from the
quality-selected checkpoints), regardless of how large that makes each
class relative to the others.

This intentionally produces class-size imbalance (e.g. micronodules ends
up much larger than emphysema) -- that's expected and NOT corrected by
subsampling here. It should be corrected downstream via per-fold
class-weighted loss (a separate, controlled follow-up step -- see notes
at the bottom of this file), not by discarding real data.

Usage:
  python build_augmented_v3_5k.py
"""

import os

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np

REAL_DIR = os.path.join(PROJECT_ROOT, "ILD_DB_npy")
SYNTH_DIR = os.path.join(PROJECT_ROOT, "diffusion_model/outputs_attention/synthetic_5k")
OUT_DIR = os.path.join(PROJECT_ROOT, "ILD_DB_npy_augmented_attention")

CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]
FIXED_DOSE = 5000  # expected synthetic count per class -- checked against actual file sizes below
SYNTHETIC_PATIENT_ID = -2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    real_images = np.load(os.path.join(REAL_DIR, "all_images.npy"))
    real_labels = np.load(os.path.join(REAL_DIR, "all_labels.npy"))
    real_pids = np.load(os.path.join(REAL_DIR, "all_patient_ids.npy"))

    all_images = [real_images]
    all_labels = [real_labels]
    all_pids = [real_pids]

    print(f"Real data (kept in full, no trimming): {len(real_images)} patches\n")

    for class_idx, class_name in enumerate(CLASS_NAMES):
        img_path = os.path.join(SYNTH_DIR, f"synthetic_{class_name}_images.npy")
        lbl_path = os.path.join(SYNTH_DIR, f"synthetic_{class_name}_labels.npy")

        if not os.path.exists(img_path):
            print(f"  [MISSING] {class_name}: no synthetic file at {img_path} -- "
                  f"did the sampling job for this class run/complete?")
            continue

        synth_images = np.load(img_path)
        synth_labels = np.load(lbl_path)

        if len(synth_images) != FIXED_DOSE:
            print(f"  [WARN] {class_name}: expected {FIXED_DOSE} synthetic patches, "
                  f"found {len(synth_images)} -- dose is not actually fixed for this class!")

        synth_pids = np.full(len(synth_images), SYNTHETIC_PATIENT_ID, dtype=np.int64)

        all_images.append(synth_images)
        all_labels.append(synth_labels)
        all_pids.append(synth_pids)

        real_n = int((real_labels == class_idx).sum())
        print(f"  {class_name:15s} real={real_n:5d}  +synthetic={len(synth_images):5d}  "
              f"total={real_n + len(synth_images):5d}")

    merged_images = np.concatenate(all_images, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    merged_pids = np.concatenate(all_pids, axis=0)

    assert len(merged_images) == len(merged_labels) == len(merged_pids), "Length mismatch after merge!"

    np.save(os.path.join(OUT_DIR, "all_images.npy"), merged_images)
    np.save(os.path.join(OUT_DIR, "all_labels.npy"), merged_labels)
    np.save(os.path.join(OUT_DIR, "all_patient_ids.npy"), merged_pids)

    print(f"\nSaved merged v3 dataset to {OUT_DIR}/")
    print(f"  Total: {len(merged_images)}  (real: {(merged_pids != SYNTHETIC_PATIENT_ID).sum()}, "
          f"synthetic: {(merged_pids == SYNTHETIC_PATIENT_ID).sum()})")

    print("\nFinal per-class totals (note the INTENTIONAL imbalance -- do not subsample this):")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        mask = merged_labels == class_idx
        total = int(mask.sum())
        print(f"  {class_name:15s} total={total:5d}")

    print("\nNEXT STEP (separate, controlled follow-up -- do not add yet to this v3 comparison):")
    print("  Add per-fold class-weighted CrossEntropyLoss to BOTH classification_lopo.py")
    print("  (baseline) and the v3 augmented script, so imbalance correction is tested as")
    print("  its own isolated variable, applied equally to both conditions.")


if __name__ == "__main__":
    main()