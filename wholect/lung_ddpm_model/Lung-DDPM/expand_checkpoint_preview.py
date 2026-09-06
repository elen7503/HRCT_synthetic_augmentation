#!/usr/bin/env python3
"""Expand a training preview into per-case condition-overlay and ROI-zoom images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--header-height", type=int, default=38)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--padding", type=int, default=48)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def overlay(image: np.ndarray, layout: np.ndarray, alpha: float) -> np.ndarray:
    result = image.astype(np.float32).copy()
    mask = np.any(layout != 0, axis=2)
    interior = mask.copy()
    interior[1:-1, 1:-1] &= (
        mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    edge = mask & ~interior
    # Thicken the contour so it remains visible without obscuring ROI texture.
    thick_edge = edge.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        thick_edge |= np.roll(edge, shift=(dy, dx), axis=(0, 1))
    thick_edge &= mask
    result[thick_edge] = (
        (1.0 - alpha) * result[thick_edge] + alpha * layout[thick_edge]
    )
    return result.round().clip(0, 255).astype(np.uint8)


def roi_crop(array: np.ndarray, mask: np.ndarray, padding: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if not len(xs):
        return array
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, array.shape[0])
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, array.shape[1])
    return array[y0:y1, x0:x1]


def add_panel(canvas: Image.Image, panel: np.ndarray, x: int, y: int, size: int) -> None:
    image = Image.fromarray(panel, mode="RGB")
    scale = min(size / image.width, size / image.height)
    image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.NEAREST if scale > 1 else Image.Resampling.LANCZOS,
    )
    left = x + (size - image.width) // 2
    top = y + (size - image.height) // 2
    canvas.paste(image, (left, top))


def main() -> None:
    args = parse_args()
    source = np.asarray(Image.open(args.input).convert("RGB"))
    size, header, gap = args.image_size, args.header_height, args.gap
    row_height = size + header
    if source.shape[1] < size * 3 + gap * 2 or source.shape[0] % row_height:
        raise ValueError(f"Unexpected preview dimensions: {source.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = source.shape[0] // row_height
    metrics = []
    for row in range(rows):
        y = row * row_height
        title_strip = source[y : y + header]
        real = source[y + header : y + header + size, :size]
        layout = source[y + header : y + header + size, size + gap : size * 2 + gap]
        generated = source[
            y + header : y + header + size,
            size * 2 + gap * 2 : size * 3 + gap * 2,
        ]
        mask = np.any(layout != 0, axis=2)
        real_gray = real.astype(np.float32).mean(axis=2)
        generated_gray = generated.astype(np.float32).mean(axis=2)
        error = np.abs(real_gray - generated_gray)
        mse_full = float(np.mean((real_gray - generated_gray) ** 2))
        mse_roi = float(np.mean((real_gray[mask] - generated_gray[mask]) ** 2))
        _, ssim_map = structural_similarity(
            real_gray,
            generated_gray,
            data_range=255,
            full=True,
        )
        row_metrics = {
            "case": row + 1,
            "roi_pixels": int(mask.sum()),
            "full": {
                "MAE": float(error.mean()),
                "PSNR": float(10 * math.log10(255**2 / mse_full)) if mse_full else None,
                "SSIM": float(ssim_map.mean()),
            },
            "roi": {
                "MAE": float(error[mask].mean()),
                "PSNR": float(10 * math.log10(255**2 / mse_roi)) if mse_roi else None,
                "SSIM": float(ssim_map[mask].mean()),
            },
        }
        metrics.append(row_metrics)
        real_overlay = overlay(real, layout, args.alpha)
        generated_overlay = overlay(generated, layout, args.alpha)
        difference = np.repeat(
            np.clip(error * 4, 0, 255).astype(np.uint8)[..., None], 3, axis=2
        )
        real_zoom = roi_crop(real_overlay, mask, args.padding)
        generated_zoom = roi_crop(generated_overlay, mask, args.padding)
        difference_zoom = roi_crop(difference, mask, args.padding)

        canvas = Image.new("RGB", (size * 3 + gap * 2, (size + header) * 2), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 12), "Real CT + condition", fill="black")
        draw.text((size + gap + 10, 12), "Generated CT + condition", fill="black")
        draw.text((size * 2 + gap * 2 + 10, 12), "Absolute difference (4x)", fill="black")
        add_panel(canvas, real_overlay, 0, header, size)
        add_panel(canvas, generated_overlay, size + gap, header, size)
        add_panel(canvas, difference, size * 2 + gap * 2, header, size)
        second_y = size + header
        draw.text((10, second_y + 12), "Real ROI zoom", fill="black")
        draw.text((size + gap + 10, second_y + 12), "Generated ROI zoom", fill="black")
        roi_text = row_metrics["roi"]
        draw.text(
            (size * 2 + gap * 2 + 10, second_y + 12),
            f"ROI difference | PSNR={roi_text['PSNR']:.2f} SSIM={roi_text['SSIM']:.3f} MAE={roi_text['MAE']:.2f}",
            fill="black",
        )
        add_panel(canvas, real_zoom, 0, second_y + header, size)
        add_panel(canvas, generated_zoom, size + gap, second_y + header, size)
        add_panel(canvas, difference_zoom, size * 2 + gap * 2, second_y + header, size)
        canvas.save(args.output_dir / f"case_{row + 1:02d}.png")

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {rows} expanded preview images in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
