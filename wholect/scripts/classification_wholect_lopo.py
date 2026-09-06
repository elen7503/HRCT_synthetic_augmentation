"""
LOPO classification on whole 512x512 CT slices (dominant-class labeled),
built from ILD_DB_wholect_npy. Real data only, no synthetic augmentation
yet.
"""

import os
import sys
import argparse
import copy

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as TF
from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Healthy", "Emphysema", "Ground Glass", "Fibrosis", "Micronodules"]
NUM_CLASSES = 5
BATCH_SIZE = 8
NUM_EPOCHS = 15
LR = 1e-3

DATA_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_npy")


class WholeCTClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
        self.features = nn.Sequential(
            block(1, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
            block(128, 128),
            block(128, 128),
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.drop(TF.relu(self.fc1(x)))
        return self.fc2(x)


def augment_batch(x):
    B = x.shape[0]
    out = []
    for i in range(B):
        img = x[i]
        k = np.random.randint(0, 4)
        img = torch.rot90(img, k, dims=[1, 2])
        if np.random.rand() < 0.5:
            img = torch.flip(img, dims=[2])  # horizontal
        if np.random.rand() < 0.5:
            img = torch.flip(img, dims=[1])  # vertical
        out.append(img)
    return torch.stack(out)


def normalize(train_imgs, test_imgs):
    img_min = min(float(train_imgs.min()), float(test_imgs.min()))
    img_max = max(float(train_imgs.max()), float(test_imgs.max()))
    denom = img_max - img_min + 1e-8
    return ((train_imgs.astype(np.float32) - img_min) / denom,
            (test_imgs.astype(np.float32) - img_min) / denom)


def run_fold(fold_idx, n_folds, patient_id, all_images, all_labels, all_pids, use_augmentation=True):
    train_mask = all_pids != patient_id
    test_mask = all_pids == patient_id

    train_imgs, test_imgs = all_images[train_mask], all_images[test_mask]
    train_lbls, test_lbls = all_labels[train_mask], all_labels[test_mask]

    if len(test_imgs) == 0:
        return None

    train_imgs, test_imgs = normalize(train_imgs, test_imgs)
    train_imgs = train_imgs[:, np.newaxis, :, :]
    test_imgs = test_imgs[:, np.newaxis, :, :]

    net = WholeCTClassifier(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_x = torch.from_numpy(train_imgs).float()
    train_y = torch.from_numpy(train_lbls).long()
    test_x = torch.from_numpy(test_imgs).float().to(DEVICE)
    test_y = torch.from_numpy(test_lbls).long().to(DEVICE)

    n_train = len(train_x)
    for epoch in range(NUM_EPOCHS):
        net.train()
        perm = torch.randperm(n_train)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = train_x[idx], train_y[idx]
            if use_augmentation:
                xb = augment_batch(xb)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_images = np.load(os.path.join(DATA_DIR, "all_images.npy"))
    all_labels = np.load(os.path.join(DATA_DIR, "all_labels.npy"))
    all_pids = np.load(os.path.join(DATA_DIR, "all_patient_ids.npy"))

    print(f"Loaded {len(all_images)} slices, {len(np.unique(all_pids))} patients")
    print(f"Augmentation: {'OFF' if args.no_augment else 'ON (rotation + flips)'}\n")

    patient_ids = sorted(np.unique(all_pids).tolist())
    n_folds = len(patient_ids)

    all_f1s, all_macro_all, all_macro_present = [], [], []
    for fold_idx, pid in enumerate(patient_ids, start=1):
        result = run_fold(fold_idx, n_folds, pid, all_images, all_labels, all_pids,
                          use_augmentation=not args.no_augment)
        if result is not None:
            f1s, macro_all, macro_present = result
            all_f1s.append(f1s)
            all_macro_all.append(macro_all)
            all_macro_present.append(macro_present)

    all_f1s = np.array(all_f1s)
    mean_f1 = all_f1s.mean(axis=0)
    std_f1 = all_f1s.std(axis=0)

    print("\n" + "=" * 60)
    print(f"Whole-CT LOPO Summary ({len(all_f1s)} folds)")
    print("=" * 60)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15} F1 = {mean_f1[i]:.4f} ± {std_f1[i]:.4f}")
    print(f"  {'Macro (5 classes)':<15} F1 = {np.mean(all_macro_all):.4f} ± {np.std(all_macro_all):.4f}")
    print(f"  {'Macro (present)':<15} F1 = {np.mean(all_macro_present):.4f} ± {np.std(all_macro_present):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()