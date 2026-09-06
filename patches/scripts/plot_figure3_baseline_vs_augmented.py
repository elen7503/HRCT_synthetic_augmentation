"""
plot_figure3_baseline_vs_augmented.py
-----------------------------------------
Figure 3 for supervisor meeting, two variants:
  3a: macro F1 only, baseline vs. augmented-v2, mean +/- std across 3 seeds
  3b: same comparison broken out per class + macro

Data is hardcoded below from the three completed seed pairs (seeds 0,1,2)
-- each value is that run's fixed-final-epoch macro/per-class F1 across
all 85 LOPO folds. Update SEED_DATA if you add more seeds later.

Usage:
  python plot_figure3_baseline_vs_augmented.py
Output:
  figure3a_macro_f1.png
  figure3b_per_class_f1.png
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Data: F1 per class per seed, for each condition
# (Healthy, Emphysema, Ground Glass, Fibrosis, Micronodules)
# ============================================================

CLASS_NAMES = ["Healthy", "Emphysema", "Ground Glass", "Fibrosis", "Micronodules"]

BASELINE = {
    0: [0.0728, 0.0478, 0.2313, 0.3511, 0.1109],
    1: [0.0742, 0.0475, 0.2204, 0.3422, 0.1119],
    2: [0.0745, 0.0376, 0.2257, 0.3496, 0.1154],
}

AUGMENTED_V2 = {
    0: [0.0739, 0.0502, 0.2402, 0.3672, 0.1124],
    1: [0.0716, 0.0505, 0.2391, 0.3594, 0.1012],
    2: [0.0768, 0.0509, 0.2275, 0.3654, 0.1039],
}

OUT_3A = "figure3a_macro_f1.png"
OUT_3B = "figure3b_per_class_f1.png"

BASELINE_COLOR = "#4C72B0"
AUGMENTED_COLOR = "#C44E52"
# ============================================================


def stack(d):
    """dict of seed -> [5 class F1s] -> array shape (n_seeds, 5)"""
    return np.array([d[s] for s in sorted(d.keys())])


def macro_per_seed(arr):
    """arr shape (n_seeds, 5) -> (n_seeds,) macro F1 per seed"""
    return arr.mean(axis=1)


def plot_figure_3a():
    base_arr = stack(BASELINE)
    aug_arr = stack(AUGMENTED_V2)

    base_macro = macro_per_seed(base_arr)   # per-seed macro F1
    aug_macro = macro_per_seed(aug_arr)

    means = [base_macro.mean(), aug_macro.mean()]
    stds = [base_macro.std(), aug_macro.std()]
    labels = ["Real only\n(baseline)", "Real + Synthetic\n(quality-selected checkpoints)"]
    colors = [BASELINE_COLOR, AUGMENTED_COLOR]

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(labels, means, yerr=stds, capsize=8, color=colors,
                  edgecolor="black", linewidth=1, width=0.55)

    # Overlay individual seed points
    for i, (arr, x) in enumerate([(base_macro, 0), (aug_macro, 1)]):
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(arr))
        ax.scatter(np.full(len(arr), x) + jitter, arr, color="black", s=30, zorder=5, alpha=0.7)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + max(stds) + 0.005,
                f"{mean:.4f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Macro F1 (mean \u00b1 std across 3 seeds)", fontsize=12)
    ax.set_title("Classification Performance: Real-Only vs.\nQuality-Corrected Augmentation",
                fontsize=13)
    ax.set_ylim(0, max(means) + max(stds) + 0.03)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_3A, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_3A}")


def plot_figure_3b():
    base_arr = stack(BASELINE)   # (3 seeds, 5 classes)
    aug_arr = stack(AUGMENTED_V2)

    base_class_mean = base_arr.mean(axis=0)
    base_class_std = base_arr.std(axis=0)
    aug_class_mean = aug_arr.mean(axis=0)
    aug_class_std = aug_arr.std(axis=0)

    base_macro = macro_per_seed(base_arr)
    aug_macro = macro_per_seed(aug_arr)

    labels = CLASS_NAMES + ["Macro"]
    base_means = list(base_class_mean) + [base_macro.mean()]
    base_stds = list(base_class_std) + [base_macro.std()]
    aug_means = list(aug_class_mean) + [aug_macro.mean()]
    aug_stds = list(aug_class_std) + [aug_macro.std()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, base_means, width, yerr=base_stds, capsize=4,
          color=BASELINE_COLOR, edgecolor="black", linewidth=0.8, label="Real only (baseline)")
    ax.bar(x + width / 2, aug_means, width, yerr=aug_stds, capsize=4,
          color=AUGMENTED_COLOR, edgecolor="black", linewidth=0.8,
          label="Real + Synthetic (quality-selected)")

    # Visually separate the "Macro" summary bar from per-class bars
    ax.axvline(len(CLASS_NAMES) - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("F1 (mean \u00b1 std across 3 seeds)", fontsize=12)
    ax.set_title("Per-Class and Macro F1: Real-Only vs. Quality-Corrected Augmentation",
                fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_3B, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_3B}")


if __name__ == "__main__":
    plot_figure_3a()
    plot_figure_3b()