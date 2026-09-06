"""
analyze_wholect_coverage.py
------------------------------
Measures, per patient, how much of ILD_DB_volumeROIs' 17-class ROI
annotation actually falls within the 5 classes used by the patch
classifier (healthy=1, emphysema=2, ground_glass=3, fibrosis=4,
micronodules=5 in the README's 1-indexed scheme).

This determines whether whole-CT classification restricted to these 5
classes has adequate data coverage, BEFORE any architecture or training
work starts.

Usage:
  python analyze_wholect_coverage.py
"""

import os
import glob

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import pydicom
from collections import defaultdict

VOLUMEROIS_DIR = os.path.join(PROJECT_ROOT, "ILD_DB/ILD_DB_volumeROIs")

# README 1-indexed label -> your 5-class scheme (0-indexed), matching
# healthy=0, emphysema=1, ground_glass=2, fibrosis=3, micronodules=4
TARGET_LABEL_MAP = {1: "healthy", 2: "emphysema", 3: "ground_glass", 4: "fibrosis", 5: "micronodules"}
ALL_LABEL_NAMES = {
    1: "healthy", 2: "emphysema", 3: "ground_glass", 4: "fibrosis", 5: "micronodules",
    6: "consolidation", 7: "bronchial_wall_thickening", 8: "reticulation",
    9: "macronodules", 10: "cysts", 11: "peripheral_micronodules", 12: "bronchiectasis",
    13: "air_trapping", 14: "early_fibrosis", 15: "increased_attenuation",
    16: "tuberculosis", 17: "pcp",
}


def analyze_patient(patient_dir):
    """Returns per-label pixel counts for one patient, across all slices."""
    roi_dir = os.path.join(patient_dir, "roi_mask")
    if not os.path.isdir(roi_dir):
        return None

    label_counts = defaultdict(int)
    n_slices = 0
    for f in glob.glob(os.path.join(roi_dir, "*.dcm")):
        try:
            arr = pydicom.dcmread(f).pixel_array
        except Exception:
            continue
        n_slices += 1
        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals, counts):
            if v != 0:  # skip background
                label_counts[int(v)] += int(c)

    return n_slices, label_counts


def main():
    patient_dirs = sorted(glob.glob(os.path.join(VOLUMEROIS_DIR, "*")))
    print(f"Found {len(patient_dirs)} patient directories\n")

    per_patient_summary = []

    for pdir in patient_dirs:
        pid = os.path.basename(pdir)
        result = analyze_patient(pdir)
        if result is None:
            continue
        n_slices, label_counts = result
        if not label_counts:
            continue

        target_counts = {name: label_counts.get(lbl, 0) for lbl, name in TARGET_LABEL_MAP.items()}
        total_target_px = sum(target_counts.values())
        total_all_px = sum(label_counts.values())
        n_target_classes_present = sum(1 for v in target_counts.values() if v > 0)

        other_labels_present = sorted(
            ALL_LABEL_NAMES.get(l, f"unknown_{l}") for l in label_counts
            if l not in TARGET_LABEL_MAP
        )

        per_patient_summary.append({
            "patient": pid,
            "n_slices_annotated": n_slices,
            "n_target_classes_present": n_target_classes_present,
            "total_target_px": total_target_px,
            "total_all_px": total_all_px,
            "pct_target_of_all": 100 * total_target_px / total_all_px if total_all_px else 0,
            "target_class_counts": target_counts,
            "other_labels_present": other_labels_present,
        })

    # ── Summary stats ────────────────────────────────────────────────────────
    n_total = len(per_patient_summary)
    n_with_any_target = sum(1 for p in per_patient_summary if p["n_target_classes_present"] >= 1)
    n_with_2plus_target = sum(1 for p in per_patient_summary if p["n_target_classes_present"] >= 2)
    n_with_only_other = n_total - n_with_any_target

    print("=" * 70)
    print(f"Total patients with any ROI annotation: {n_total}")
    print(f"Patients with >=1 of your 5 target classes present: {n_with_any_target}")
    print(f"Patients with >=2 target classes present (needed for 'top-2 by %'): {n_with_2plus_target}")
    print(f"Patients with ONLY non-target classes (consolidation, etc.): {n_with_only_other}")
    print("=" * 70)

    print("\nPer-target-class: how many patients have ANY pixels of that class?")
    for name in TARGET_LABEL_MAP.values():
        n_patients_with_class = sum(1 for p in per_patient_summary if p["target_class_counts"][name] > 0)
        print(f"  {name:15s}: {n_patients_with_class} patients")

    print("\nDistribution of n_target_classes_present per patient:")
    from collections import Counter
    dist = Counter(p["n_target_classes_present"] for p in per_patient_summary)
    for k in sorted(dist.keys()):
        print(f"  {k} target classes present: {dist[k]} patients")

    print("\nSample of patients with 2+ target classes (first 10):")
    for p in [p for p in per_patient_summary if p["n_target_classes_present"] >= 2][:10]:
        nonzero = {k: v for k, v in p["target_class_counts"].items() if v > 0}
        print(f"  {p['patient']}: {nonzero}")

    import json
    with open("wholect_coverage_analysis.json", "w") as f:
        json.dump(per_patient_summary, f, indent=2)
    print("\nFull per-patient data saved to wholect_coverage_analysis.json")


if __name__ == "__main__":
    main()