"""
build_wholect_fixed_split.py
--------------------------------
Creates a fixed, stratified 80/20 patient-level train/test split for the
mean/max-pooling patient-level classification experiment.

Also cross-checks the test patients against Liwei's train_manifest.json
(the patients his V3 DDPM checkpoint was trained on).

Usage:
  python build_wholect_fixed_split.py --seed 0
"""

import os
import json
import argparse

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_uint8")
TRAIN_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "wholect/lung_ddpm_model/Lung-DDPM/checkpoints/ILD-DDPM-2D-V3/train_manifest.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_uint8")
CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    all_labels = np.load(os.path.join(REAL_DATA_DIR, "all_labels.npy"))
    all_pids = np.load(os.path.join(REAL_DATA_DIR, "all_patient_ids.npy"))

    patient_ids = np.unique(all_pids)
    patient_labels = np.array([
        int(stats.mode(all_labels[all_pids == pid], keepdims=False).mode)
        for pid in patient_ids
    ])

    print(f"Total patients: {len(patient_ids)}")
    print("Per-class patient counts:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:15s}: {(patient_labels == c).sum()} patients")

    with open(TRAIN_MANIFEST_PATH) as f:
        liwei_train_manifest = json.load(f)
    liwei_train_patient_ids = set()
    for record in liwei_train_manifest["records"]:
        try:
            liwei_train_patient_ids.add(int(record["patient_id"]))
        except ValueError:
            pass

    clean_mask = np.array([int(pid) not in liwei_train_patient_ids for pid in patient_ids])
    clean_pids = patient_ids[clean_mask]
    clean_labels = patient_labels[clean_mask]

    print(f"Patients NOT seen by Liwei generator (clean test candidates): {len(clean_pids)} / {len(patient_ids)}")
    print("Per-class clean-patient counts:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:15s}: {(clean_labels == c).sum()} patients")

    n_test = max(1, int(round(len(patient_ids) * args.test_size)))
    n_test = min(n_test, len(clean_pids))

    from collections import Counter
    class_counts = Counter(clean_labels.tolist())
    can_stratify = all(c >= 2 for c in class_counts.values()) and len(class_counts) > 1

    if can_stratify:
        _, test_pids = train_test_split(
            clean_pids, test_size=n_test, stratify=clean_labels, random_state=args.seed
        )
    else:
        rng = np.random.default_rng(args.seed)
        test_pids = rng.choice(clean_pids, size=n_test, replace=False)
        print("  [WARN] Could not stratify -- used unstratified random sampling from clean pool.")

    train_pids = np.array([pid for pid in patient_ids if pid not in test_pids])

    print(f"\nSplit: {len(train_pids)} train patients, {len(test_pids)} test patients")
    print("\nTest set per-class counts:")
    for c, name in enumerate(CLASS_NAMES):
        n = sum(1 for pid in test_pids if patient_labels[list(patient_ids).index(pid)] == c)
        print(f"  {name:15s}: {n} patients")

    with open(TRAIN_MANIFEST_PATH) as f:
        liwei_train_manifest = json.load(f)
    liwei_train_patient_ids = set()
    for record in liwei_train_manifest["records"]:
        try:
            liwei_train_patient_ids.add(int(record["patient_id"]))
        except ValueError:
            pass

    overlap = [int(pid) for pid in test_pids if int(pid) in liwei_train_patient_ids]
    print(f"\nTest patients that WERE used to train Liwei's V3 DDPM checkpoint: {len(overlap)} / {len(test_pids)}")
    if overlap:
        print(f"  Overlapping patient IDs: {overlap}")
        print("  CAVEAT: worth reporting as a limitation.")
    else:
        print("  None -- test set is fully unseen by both classifier and generator training.")

    np.save(os.path.join(OUT_DIR, f"fixed_split_train_pids_seed{args.seed}.npy"), train_pids)
    np.save(os.path.join(OUT_DIR, f"fixed_split_test_pids_seed{args.seed}.npy"), test_pids)
    print(f"\nSaved split to {OUT_DIR}/fixed_split_{{train,test}}_pids_seed{args.seed}.npy")


if __name__ == "__main__":
    main()
