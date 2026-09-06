"""
plot_figure1_recall_vs_epoch.py
----------------------------------
Figure 1 for supervisor meeting: shows recall (sample diversity) as a
function of training checkpoint epoch, per class -- the core finding
that "lowest training loss" (best.pt) does not correspond to "best
sample quality" for most classes.

Usage:
  python plot_figure1_recall_vs_epoch.py
Output:
  figure1_recall_vs_epoch.png
"""

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
CSV_PATH = "/rds/general/user/eh1121/home/Final_Project/checkpoint_sweep_all_classes_results.csv"  # EDIT if not in current directory
OUT_PATH = "figure1_recall_vs_epoch.png"

CLASS_COLORS = {
    "healthy": "#4C72B0",
    "emphysema": "#DD8452",
    "ground_glass": "#55A868",
    "fibrosis": "#C44E52",
    "micronodules": "#8172B2",
}
# ============================================================


def main():
    df = pd.read_csv(CSV_PATH)

    fig, ax = plt.subplots(figsize=(11, 6))

    for class_name, group in df.groupby("class_name"):
        group = group.sort_values("epoch")
        color = CLASS_COLORS.get(class_name, None)

        # Periodic checkpoints: line + circle markers
        for class_name, group in df.groupby("class_name"):
            group = group.sort_values("epoch")  # includes best.pt now, at its own epoch
            color = CLASS_COLORS.get(class_name, None)

    # One continuous line through ALL checkpoints, periodic + best, sorted by epoch
            ax.plot(group["epoch"], group["recall"], marker="o", markersize=5,
                linewidth=2, color=color, label=class_name.replace("_", " ").title())

    # Re-draw the best.pt point on top with a star marker, same position on the line
            best_row = group[group["checkpoint"] == "best"]
            if not best_row.empty:
                ax.scatter(best_row["epoch"], best_row["recall"], marker="*", s=350,
                    color=color, edgecolor="black", linewidth=1.2, zorder=5)
        
        best_row = group[group["checkpoint"] == "best"]
        if not best_row.empty:
            ax.scatter(best_row["epoch"], best_row["recall"], marker="*", s=350,
                       color=color, edgecolor="black", linewidth=1.2, zorder=5)

    # One manual legend entry to explain the star marker
    from matplotlib.lines import Line2D
    star_legend = Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                          markeredgecolor="black", markersize=16,
                          label="best.pt (selected by lowest training loss)")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(star_legend)
    labels.append("best.pt (selected by lowest training loss)")

    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Recall (sample diversity vs. real data)", fontsize=12)
    ax.set_title("Sample Diversity Declines With Training Duration,\n"
                 "While the Loss-Selected Checkpoint (\u2605) Is Often Far From Optimal",
                 fontsize=13)
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=1.0)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()