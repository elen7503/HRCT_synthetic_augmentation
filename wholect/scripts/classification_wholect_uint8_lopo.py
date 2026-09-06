"""
classification_wholect_uint8_lopo.py
----------------------------------------
Same LOPO classification protocol as classification_wholect_lopo.py, but
for uint8 lung-window images (0-255) instead of raw HU -- used for both
the real-only and real+synthetic augmented whole-CT comparisons, so both
runs use IDENTICAL preprocessing and only differ in what's in DATA_DIR.

Usage:
  python classification_wholect_uint8_lopo.py --data-dir /path/to/dataset --seed 0
"""

import os
import argparse

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from classification_wholect_lopo import WholeCTClassifier, NUM_CLASSES, CLASS_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 15
LR = 1e-3


def run_fold(fold_idx, n_folds, patient_id, all_images, all_labels, all_pids):
    train_mask = all_pids != patient_id
    test_mask = all_pids == patient_id

    train_imgs, test_imgs = all_images[train_mask], all_images[test_mask]
    train_lbls, test_lbls = all_labels[train_mask], all_labels[test_mask]

    if len(test_imgs) == 0:
        return None

    train_x = torch.from_numpy(train_imgs.astype(np.float32) / 255.0).unsqueeze(1)
    train_y = torch.from_numpy(train_lbls).long()
    test_x = torch.from_numpy(test_imgs.astype(np.float32) / 255.0).unsqueeze(1).to(DEVICE)
    test_y = torch.from_numpy(test_lbls).long().to(DEVICE)

    net = WholeCTClassifier(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    n_train = len(train_x)
    for epoch in range(NUM_EPOCHS):
        net.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = train_x[idx].to(DEVICE), train_y[idx].to(DEVICE)
            optimizer.zero_grad()
            out = net(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    net.eval()
    with torch.no_grad():
        preds = net(test_x).argmax(dim=1).cpu().numpy()
    true = test_y.cpu().numpy()

    f1_per_class = f1_score(true, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    present_classes = sorted(set(true.tolist()))
    macro_f1_present = f1_score(true, preds, labels=present_classes, average="macro", zero_division=0)

    print(f"Fold {fold_idx:3d}/{n_folds} | patient={patient_id} | train={n_train} test={len(test_x)} | "
          f"F1: " + " | ".join(f"{CLASS_NAMES[i]}={f1_per_class[i]:.3f}" for i in range(NUM_CLASSES)) +
          f" | macro(present)={macro_f1_present:.3f} present={present_classes}")

    return f1_per_class, float(np.mean(f1_per_class)), macro_f1_present


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_images = np.load(os.path.join(args.data_dir, "all_images.npy"))
    all_labels = np.load(os.path.join(args.data_dir, "all_labels.npy"))
    all_pids = np.load(os.path.join(args.data_dir, "all_patient_ids.npy"))

    print(f"Loaded {len(all_images)} slices from {args.data_dir}")
    print(f"Distinct patient IDs: {len(np.unique(all_pids))}\n")

    REAL_PIDS_SOURCE = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_uint8/all_patient_ids.npy")
    real_pids_fixed = np.load(REAL_PIDS_SOURCE)
    real_test_candidates = sorted(np.unique(real_pids_fixed).tolist())
    n_folds = len(real_test_candidates)

    all_f1s, all_macro_present = [], []
    for fold_idx, pid in enumerate(real_test_candidates, start=1):
        result = run_fold(fold_idx, n_folds, pid, all_images, all_labels, all_pids)
        if result is not None:
            f1s, macro_all, macro_present = result
            all_f1s.append(f1s)
            all_macro_present.append(macro_present)

    all_f1s = np.array(all_f1s)
    mean_f1 = all_f1s.mean(axis=0)
    std_f1 = all_f1s.std(axis=0)

    print("\n" + "=" * 60)
    print(f"Whole-CT LOPO Summary ({len(all_f1s)} folds) -- {args.data_dir}")
    print("=" * 60)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15} F1 = {mean_f1[i]:.4f} ± {std_f1[i]:.4f}")
    print(f"  {'Macro (5 classes)':<15} F1 = {np.mean(all_f1s.mean(axis=1)):.4f} ± {np.std(all_f1s.mean(axis=1)):.4f}")
    print(f"  {'Macro (present)':<15} F1 = {np.mean(all_macro_present):.4f} ± {np.std(all_macro_present):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
