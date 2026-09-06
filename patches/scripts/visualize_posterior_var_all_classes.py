"""
Generates one PNG per class comparing real patches against the NEW
synthetic samples (produced with the corrected posterior-variance
sampling formula), saved separately from the original augmented dataset
so this is a clean before/after check.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

REAL_IMGS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy/all_images.npy"
REAL_LBLS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy/all_labels.npy"

SYNTH_DIR = "/rds/general/user/eh1121/home/Final_Project/diffusion_model/outputs/synthetic_test_posterior_var"
OUTPUT_DIR = SYNTH_DIR  # save grids alongside the samples

CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]
N_SHOW = 10
SEED = 0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    real_images = np.load(REAL_IMGS_PATH)
    real_labels = np.load(REAL_LBLS_PATH)

    for class_idx, class_name in enumerate(CLASS_NAMES):
        synth_path = os.path.join(SYNTH_DIR, f"synthetic_{class_name}_images.npy")
        if not os.path.exists(synth_path):
            print(f"[SKIP] {class_name}: no file at {synth_path} (checkpoint may not exist for this class)")
            continue

        synth_images = np.load(synth_path)
        real_idx = np.where(real_labels == class_idx)[0]
        n_real_show = min(N_SHOW, len(real_idx))
        n_synth_show = min(N_SHOW, len(synth_images))
        real_pick = rng.choice(real_idx, size=n_real_show, replace=False)

        fig, axes = plt.subplots(2, N_SHOW, figsize=(N_SHOW * 1.2, 2.8))
        for col in range(N_SHOW):
            ax = axes[0, col]
            if col < n_real_show:
                ax.imshow(real_images[real_pick[col]], cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_title("REAL", loc="left", fontsize=9)

            ax = axes[1, col]
            if col < n_synth_show:
                ax.imshow(synth_images[col], cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_title("SYNTHETIC (posterior var)", loc="left", fontsize=9)

        fig.suptitle(f"{class_name}  |  real n={len(real_idx)}  synthetic (test) n={len(synth_images)}", fontsize=11)
        fig.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"comparison_{class_name}_posterior_var.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()