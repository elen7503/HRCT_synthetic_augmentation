"""
plot_figure2_all_classes_comparison.py
------------------------------------------
Figure 2 for supervisor meeting, generalized to all 5 classes: for each
class, a three-row grid -- REAL, SYNTHETIC (best.pt, lowest training
loss), SYNTHETIC (quality-selected checkpoint from the sweep). Row
labels are placed as text ABOVE each row rather than to the left.
 
Saves one PNG per class: figure2_healthy_comparison.png,
figure2_emphysema_comparison.png, etc.
 
Usage:
  python plot_figure2_all_classes_comparison.py
"""
 
import os
import numpy as np
import matplotlib.pyplot as plt
 
# ============================================================
REAL_IMGS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy/all_images.npy"
REAL_LBLS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy/all_labels.npy"
 
BEST_PT_SYNTH_DIR = "/rds/general/user/eh1121/home/Final_Project/diffusion_model/outputs/synthetic_test_posterior_var"
SWEEP_DIR = "/rds/general/user/eh1121/home/Final_Project/diffusion_model/outputs/checkpoint_sweep"
 
OUT_DIR = "."  # where to save the PNGs
N_SHOW = 8
SEED = 0
 
# class_idx: (class_name, quality-selected checkpoint folder name)
CLASS_CONFIG = {
    0: ("healthy", "epoch_0300"),
    1: ("emphysema", "epoch_2000"),
    2: ("ground_glass", "epoch_0200"),
    3: ("fibrosis", "epoch_0100"),
    4: ("micronodules", "epoch_0100"),
}
# ============================================================
 
 
def make_figure(class_idx, class_name, best_epoch_folder, real_images, real_labels, rng):
    real_patches = real_images[real_labels == class_idx]
 
    best_pt_path = os.path.join(BEST_PT_SYNTH_DIR, f"synthetic_{class_name}_images.npy")
    quality_path = os.path.join(SWEEP_DIR, class_name, best_epoch_folder, "synthetic_images.npy")
 
    if not os.path.exists(best_pt_path):
        print(f"[SKIP] {class_name}: missing {best_pt_path}")
        return
    if not os.path.exists(quality_path):
        print(f"[SKIP] {class_name}: missing {quality_path}")
        return
 
    best_pt_synth = np.load(best_pt_path)
    quality_synth = np.load(quality_path)
 
    n_real = min(N_SHOW, len(real_patches))
    n_best = min(N_SHOW, len(best_pt_synth))
    n_quality = min(N_SHOW, len(quality_synth))
 
    real_pick = rng.choice(len(real_patches), size=n_real, replace=False)
    best_pick = rng.choice(len(best_pt_synth), size=n_best, replace=False)
    quality_pick = rng.choice(len(quality_synth), size=n_quality, replace=False)
 
    fig, axes = plt.subplots(3, N_SHOW, figsize=(N_SHOW * 1.4, 3 * 1.7))
 
    row_data = [
        ("REAL", real_patches, real_pick),
        ("SYNTHETIC \u2014 best.pt (lowest training loss)", best_pt_synth, best_pick),
        (f"SYNTHETIC \u2014 {best_epoch_folder.replace('_', ' ')} (quality-selected)", quality_synth, quality_pick),
    ]
 
    for row, (row_label, data, pick) in enumerate(row_data):
        for col in range(N_SHOW):
            ax = axes[row, col]
            if col < len(pick):
                ax.imshow(data[pick[col]], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
 
        # Row label as text ABOVE the row (above its leftmost column), not to the left
        axes[row, 0].text(0, -0.15, row_label, transform=axes[row, 0].transAxes,
                          fontsize=10, fontweight="bold", ha="left", va="bottom")
 
    fig.suptitle(f"{class_name.replace('_', ' ').title()}: Real vs. Synthetic Patches",
                fontsize=14, y=1.04)
    fig.tight_layout()
 
    out_path = os.path.join(OUT_DIR, f"figure2_{class_name}_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
 
 
def main():
    rng = np.random.default_rng(SEED)
    real_images = np.load(REAL_IMGS_PATH)
    real_labels = np.load(REAL_LBLS_PATH)
 
    for class_idx, (class_name, best_epoch_folder) in CLASS_CONFIG.items():
        make_figure(class_idx, class_name, best_epoch_folder, real_images, real_labels, rng)
 
 
if __name__ == "__main__":
    main()
 