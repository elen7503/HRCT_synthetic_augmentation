"""
visualize_real_vs_synthetic.py
--------------------------------
Generates one PNG per class: a grid of real patches on top, synthetic
patches (from the corresponding DDPM) below, for direct visual comparison.

Purpose: answer "do the generated samples look visually reasonable" with
actual evidence, not just a claim -- useful for supervisor communication
and as a thesis figure.

Usage:
  python visualize_real_vs_synthetic.py
Output:
  comparison_healthy.png, comparison_emphysema.png, ... in OUTPUT_DIR
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG -- edit if your paths differ
# ============================================================
REAL_IMGS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy/all_images.npy"
REAL_LBLS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy/all_labels.npy"

AUG_IMGS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/all_images.npy"
AUG_LBLS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/all_labels.npy"
AUG_PIDS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/all_patient_ids.npy"

CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]
N_SAMPLES_PER_ROW = 10   # how many example patches to show per row
SEED = 0

OUTPUT_DIR = "/rds/general/user/eh1121/home/Final_Project/diffusion_model/outputs/real_vs_synth_grids"

# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    real_images = np.load(REAL_IMGS_PATH)
    real_labels = np.load(REAL_LBLS_PATH)

    aug_images = np.load(AUG_IMGS_PATH)
    aug_labels = np.load(AUG_LBLS_PATH)
    aug_pids = np.load(AUG_PIDS_PATH)
    synth_mask_all = aug_pids == -2

    for class_idx, class_name in enumerate(CLASS_NAMES):
        real_idx = np.where(real_labels == class_idx)[0]
        synth_idx = np.where(synth_mask_all & (aug_labels == class_idx))[0]

        n_real_show = min(N_SAMPLES_PER_ROW, len(real_idx))
        n_synth_show = min(N_SAMPLES_PER_ROW, len(synth_idx))

        real_pick = rng.choice(real_idx, size=n_real_show, replace=False) if n_real_show > 0 else []
        synth_pick = rng.choice(synth_idx, size=n_synth_show, replace=False) if n_synth_show > 0 else []

        n_rows = 2 if n_synth_show > 0 else 1
        fig, axes = plt.subplots(n_rows, N_SAMPLES_PER_ROW, figsize=(N_SAMPLES_PER_ROW * 1.2, n_rows * 1.4))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for col in range(N_SAMPLES_PER_ROW):
            ax = axes[0, col]
            if col < n_real_show:
                ax.imshow(real_images[real_pick[col]], cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_title("REAL", loc="left", fontsize=9)

        if n_rows == 2:
            for col in range(N_SAMPLES_PER_ROW):
                ax = axes[1, col]
                if col < n_synth_show:
                    ax.imshow(synth_images_patch(aug_images, synth_pick, col), cmap="gray")
                ax.axis("off")
                if col == 0:
                    ax.set_title("SYNTHETIC", loc="left", fontsize=9)

        note = "" if n_synth_show > 0 else "  (NO SYNTHETIC PATCHES EXIST FOR THIS CLASS)"
        fig.suptitle(f"{class_name}  |  real n={len(real_idx)}  synthetic n={len(synth_idx)}{note}", fontsize=11)
        fig.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"comparison_{class_name}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def synth_images_patch(aug_images, synth_pick, col):
    return aug_images[synth_pick[col]]


if __name__ == "__main__":
    main()