"""
Trains ONE WholeCTClassifier on ALL real whole-CT data (no LOPO held-out
split.
"""

import os

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from classification_wholect_lopo import WholeCTClassifier, NUM_CLASSES, DEVICE

DATA_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_npy")
OUT_PATH = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_npy/feature_extractor.pth")
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-3


def apply_lung_window(hu_image, center=-600, width=1600):
    lo = center - width / 2
    hi = center + width / 2
    clipped = np.clip(hu_image, lo, hi).astype(np.float32)
    return (clipped - lo) / (hi - lo)  # -> [0, 1] directly


def main():
    all_images = np.load(os.path.join(DATA_DIR, "all_images.npy"))
    all_labels = np.load(os.path.join(DATA_DIR, "all_labels.npy"))

    imgs_norm = apply_lung_window(all_images)
    imgs_norm = imgs_norm[:, np.newaxis, :, :]

    x = torch.from_numpy(imgs_norm).float()
    y = torch.from_numpy(all_labels).long()

    net = WholeCTClassifier(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    n = len(x)
    print(f"Training feature extractor on {n} real whole-CT slices, {EPOCHS} epochs")

    for epoch in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = x[idx].to(DEVICE), y[idx].to(DEVICE)
            optimizer.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch:2d}/{EPOCHS} | loss: {epoch_loss / n_batches:.4f}")

    torch.save({"state_dict": net.state_dict()}, OUT_PATH)
    print(f"\nSaved feature extractor to {OUT_PATH}")


if __name__ == "__main__":
    main()