"""
wilcoxon_baseline_vs_v3_5k.py

One-sided Wilcoxon signed-rank test (alternative='greater') comparing
per-fold macro F1 between the baseline (no synthetic augmentation) and
the fixed-dose v3_5k (5000 synthetic patches/class) LOPO conditions,
patch-level classification pipeline (Branch A).

Pairing: fold i of condition X is the same held-out patient as fold i
of condition Y, for a given seed (LOPO folds are generated in a fixed,
deterministic patient order, verified below before pairing).

Data source: per-fold "Macro-F1 (5 fixed classes)" lines in the LOPO
training logs (see PROJECT_STATUS.md, Branch A key file locations).
The v3_5k logs used here are the CORRECTED/"FIXED" reruns (10 Aug 2026)
that include synthetic patches with patient_id=-2 in the training mask
(see commit 887c23c) -- these supersede the earlier v3_5k run referenced
as "not in experiments/" in PROJECT_STATUS.md.

Usage:
  python wilcoxon_baseline_vs_v3_5k.py
"""

import re
import sys
from pathlib import Path

import os
PROJECT_ROOT = os.environ["PROJECT_ROOT"]

import numpy as np
from scipy.stats import wilcoxon

REPO_ROOT = Path(PROJECT_ROOT)
PATCHES = REPO_ROOT / "patches"

# (seed, baseline_log, augmented_v3_5k_log)
CONDITIONS = [
    (0, PATCHES / "logs" / "lopo_baseline.out",
        PATCHES / "experiments" / "lopo_augmented_v3_5k_seed0_20260810_082317" / "train.log"),
    (1, PATCHES / "experiments" / "lopo_baseline_seed1_20260728_115856" / "train.log",
        PATCHES / "experiments" / "lopo_augmented_v3_5k_seed1_20260810_082317" / "train.log"),
    (2, PATCHES / "experiments" / "lopo_baseline_seed2_20260728_121449" / "train.log",
        PATCHES / "experiments" / "lopo_augmented_v3_5k_seed2_20260810_093654" / "train.log"),
]

FOLD_RE = re.compile(r"^Fold\s+\d+/\d+\s+\|\s+patient_id=(\d+)")
F1_RE = re.compile(r"Macro-F1 \(5 fixed classes\):\s*([\d.]+)")


def load_fold_results(log_path):
    """Return (patient_ids, f1_macro) arrays, in fold order, from a LOPO train.log/.out file."""
    patient_ids, f1_values = [], []
    pending_pid = None
    with open(log_path) as fh:
        for line in fh:
            m_fold = FOLD_RE.match(line)
            if m_fold:
                pending_pid = int(m_fold.group(1))
                continue
            m_f1 = F1_RE.search(line)
            if m_f1:
                if pending_pid is None:
                    raise ValueError(f"{log_path}: found Macro-F1 line with no preceding Fold header")
                patient_ids.append(pending_pid)
                f1_values.append(float(m_f1.group(1)))
                pending_pid = None
    if not patient_ids:
        raise ValueError(f"{log_path}: no per-fold Macro-F1 results found")
    return np.array(patient_ids), np.array(f1_values)


def main():
    per_seed = {}
    all_baseline, all_augmented = [], []

    print("=" * 70)
    print("Wilcoxon signed-rank test: v3_5k (fixed-dose, 5000/class) "
          "vs baseline")
    print("Patch-level LOPO classification, per-fold macro F1 (5 classes)")
    print("H1 (alternative='greater'): augmented F1 > baseline F1")
    print("=" * 70)

    for seed, baseline_log, augmented_log in CONDITIONS:
        pid_base, f1_base = load_fold_results(baseline_log)
        pid_aug, f1_aug = load_fold_results(augmented_log)

        if pid_base.shape != pid_aug.shape:
            sys.exit(f"[seed {seed}] fold count mismatch: "
                      f"baseline={len(pid_base)} augmented={len(pid_aug)}")
        if not np.array_equal(pid_base, pid_aug):
            mismatch = np.where(pid_base != pid_aug)[0]
            sys.exit(f"[seed {seed}] patient order mismatch at fold index(es) "
                      f"{mismatch.tolist()} -- refusing to pair misaligned folds")

        n = len(pid_base)
        stat, p = wilcoxon(f1_aug, f1_base, alternative="greater")

        per_seed[seed] = (f1_base, f1_aug)
        all_baseline.append(f1_base)
        all_augmented.append(f1_aug)

        print(f"\nSeed {seed}  ({n} folds, alignment verified: same "
              f"patient_id order in both logs)")
        print(f"  baseline mean F1  = {f1_base.mean():.4f}")
        print(f"  augmented mean F1 = {f1_aug.mean():.4f}")
        print(f"  Wilcoxon W = {stat:.4f}   p-value = {p:.6g}")

    pooled_baseline = np.concatenate(all_baseline)
    pooled_augmented = np.concatenate(all_augmented)
    stat_pooled, p_pooled = wilcoxon(pooled_augmented, pooled_baseline, alternative="greater")

    print("\n" + "=" * 70)
    print(f"Pooled across all 3 seeds ({len(pooled_baseline)} paired folds)")
    print(f"  baseline mean F1  = {pooled_baseline.mean():.4f}")
    print(f"  augmented mean F1 = {pooled_augmented.mean():.4f}")
    print(f"  Wilcoxon W = {stat_pooled:.4f}   p-value = {p_pooled:.6g}")
    print("=" * 70)

    # Average F1 per fold across the 3 seeds first (noise reduction), then a
    # single Wilcoxon test over the resulting 85 fold-averaged pairs. Folds
    # are in the same patient_id order across seeds (verified above via the
    # per-seed alignment check), so averaging along axis=0 is safe.
    baseline_stack = np.stack(all_baseline, axis=0)   # (3 seeds, 85 folds)
    augmented_stack = np.stack(all_augmented, axis=0)
    baseline_avg = baseline_stack.mean(axis=0)        # (85,)
    augmented_avg = augmented_stack.mean(axis=0)
    stat_avg, p_avg = wilcoxon(augmented_avg, baseline_avg, alternative="greater")

    print("\n" + "=" * 70)
    print(f"Seed-averaged (mean F1 per fold across 3 seeds, {len(baseline_avg)} folds)")
    print(f"  baseline mean F1  = {baseline_avg.mean():.4f}")
    print(f"  augmented mean F1 = {augmented_avg.mean():.4f}")
    print(f"  Wilcoxon W = {stat_avg:.4f}   p-value = {p_avg:.6g}")
    print("=" * 70)


if __name__ == "__main__":
    main()
