"""
Trains the feature extractor on Liwei's own training images
(train_manifest.json), not our independently-built ILD_DB_wholect_npy.
"""

import os
import json

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from classification_wholect_lopo import WholeCTClassifier, NUM_CLASSES, DEVICE

ILD_DATASET_DIR = os.path.join(PROJECT_ROOT, "wholect/lung_ddpm_model/ILD_dataset")
TRAIN_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "wholect/lung_ddpm_model/Lung-DDPM/checkpoints/ILD-DDPM-2D-V3/train_manifest.json")
OUT_PATH = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_npy/feature_extractor_lungddpm_domain.pth")

LUNGDDPM_LABEL_TO_YOUR_CLASS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-3


def main():
    with open(TRAIN_MANIFEST_PATH) as f:
        manifest = json.load(f)

    images, labels = [], []
    for record in manifest["records"]:
        target_class = None
        for lbl in record["labels"]:
            if lbl in LUNGDDPM_LABEL_TO_YOUR_CLASS:
                target_class = LUNGDDPM_LABEL_TO_YOUR_CLASS[lbl]
                break
        if target_class is None:
            continue
        img_path = os.path.join(ILD_DATASET_DIR, record["files"]["image"]["path"])
        if not os.path.exists(img_path):
            continue
        arr = np.load(img_path)
        if arr.shape != (512, 512):
            import cv2
            arr = cv2.resize(arr.astype(np.float32), (512, 512), interpolation=cv2.INTER_LINEAR)
        images.append(arr)
        labels.append(target_class)

    images = np.stack(images).astype(np.float32) / 255.0
    labels = np.array(labels, dtype=np.int64)
    print(f"Loaded {len(images)} of Liwei's own training slices across {len(np.unique(labels))} classes")
    for c in range(NUM_CLASSES):
        print(f"  class {c}: {(labels == c).sum()} slices")

    x = torch.from_numpy(images).unsqueeze(1)
    y = torch.from_numpy(labels)

    net = WholeCTClassifier(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    n = len(x)
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
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
