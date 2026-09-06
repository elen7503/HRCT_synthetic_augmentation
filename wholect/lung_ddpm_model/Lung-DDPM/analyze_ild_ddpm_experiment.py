#!/usr/bin/env python3
"""Aggregate checkpoint-preview diagnostics for an ILD DDPM experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


LABEL_NAMES = {
    1: "healthy_control",
    2: "emphysema",
    3: "ground_glass",
    4: "fibrosis",
    5: "micronodules",
    6: "consolidation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiment = args.experiment_dir.resolve()
    output = (args.output_dir or experiment / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    metric_files = sorted((experiment / "preview_metrics").glob("metrics_step_*.json"))
    if not metric_files:
        raise FileNotFoundError(f"No preview metrics found in {experiment}")

    summary_rows: list[dict] = []
    case_rows: list[dict] = []
    label_rows: list[dict] = []
    for metric_file in metric_files:
        payload = json.loads(metric_file.read_text(encoding="utf-8"))
        step = int(payload["step"])
        cases = payload["cases"]
        roi = [case["roi"] for case in cases if case["roi"]["pixels"]]
        total_pixels = sum(int(item["pixels"]) for item in roi)
        weighted_mse = sum(
            int(item["pixels"]) * 255**2 / (10 ** (float(item["PSNR"]) / 10))
            for item in roi
            if item["PSNR"] is not None
        ) / total_pixels
        summary_rows.append(
            {
                "step": step,
                "num_cases": len(roi),
                "roi_pixels": total_pixels,
                "mean_roi_mae": mean([float(item["MAE"]) for item in roi]),
                "mean_roi_psnr": mean([float(item["PSNR"]) for item in roi]),
                "mean_roi_ssim": mean([float(item["SSIM"]) for item in roi]),
                "pixel_weighted_roi_mae": sum(
                    int(item["pixels"]) * float(item["MAE"]) for item in roi
                ) / total_pixels,
                "pixel_weighted_roi_psnr": 10 * math.log10(255**2 / weighted_mse),
                "pixel_weighted_roi_ssim": sum(
                    int(item["pixels"]) * float(item["SSIM"]) for item in roi
                ) / total_pixels,
            }
        )
        per_label: dict[int, list[dict]] = defaultdict(list)
        for case in cases:
            row = {
                "step": step,
                "patient_id": case["patient_id"],
                "slice_id": case["slice_id"],
                "labels": "+".join(str(label) for label in case["labels"]),
                "roi_pixels": case["roi"]["pixels"],
                "roi_mae": case["roi"]["MAE"],
                "roi_psnr": case["roi"]["PSNR"],
                "roi_ssim": case["roi"]["SSIM"],
            }
            case_rows.append(row)
            for label in case["labels"]:
                per_label[int(label)].append(case["roi"])
        for label, items in sorted(per_label.items()):
            label_rows.append(
                {
                    "step": step,
                    "label": label,
                    "label_name": LABEL_NAMES.get(label, "unknown"),
                    "num_cases": len(items),
                    "mean_roi_mae": mean([float(item["MAE"]) for item in items]),
                    "mean_roi_psnr": mean([float(item["PSNR"]) for item in items]),
                    "mean_roi_ssim": mean([float(item["SSIM"]) for item in items]),
                }
            )

    write_csv(output / "checkpoint_summary.csv", summary_rows)
    write_csv(output / "per_case_metrics.csv", case_rows)
    write_csv(output / "per_label_metrics.csv", label_rows)

    best_ssim = max(summary_rows, key=lambda row: row["mean_roi_ssim"])
    best_psnr = max(summary_rows, key=lambda row: row["mean_roi_psnr"])
    best_mae = min(summary_rows, key=lambda row: row["mean_roi_mae"])
    report = {
        "experiment": str(experiment),
        "warning": (
            "PSNR, SSIM and MAE compare stochastic samples with one real target. "
            "They are checkpoint diagnostics, not standalone realism or medical-validity scores."
        ),
        "num_checkpoints": len(summary_rows),
        "num_fixed_validation_cases": summary_rows[0]["num_cases"],
        "best_mean_roi_ssim_step": best_ssim["step"],
        "best_mean_roi_psnr_step": best_psnr["step"],
        "best_mean_roi_mae_step": best_mae["step"],
    }
    (output / "analysis_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    try:
        import matplotlib.pyplot as plt

        steps = [row["step"] for row in summary_rows]
        figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        for axis, key, label in (
            (axes[0], "mean_roi_psnr", "Mean paired ROI PSNR"),
            (axes[1], "mean_roi_ssim", "Mean paired ROI SSIM"),
            (axes[2], "mean_roi_mae", "Mean paired ROI MAE"),
        ):
            axis.plot(steps, [row[key] for row in summary_rows], marker="o")
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)
        axes[-1].set_xlabel("Training step")
        figure.suptitle("Fixed validation-sample diagnostics (not realism scores)")
        figure.tight_layout()
        figure.savefig(output / "checkpoint_metric_curves.png", dpi=180)
        plt.close(figure)
    except ImportError:
        pass

    print(json.dumps(report, indent=2))
    print(f"Analysis written to: {output}")


if __name__ == "__main__":
    main()
