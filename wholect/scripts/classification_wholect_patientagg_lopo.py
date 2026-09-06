"""
Same LOPO training as classification_wholect_uint8_lopo.py, but
evaluation aggregates predictions to PATIENT level:.
"""

import os
import argparse

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
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
        slice_preds = net(test_x).argmax(dim=1).cpu().numpy()

    patient_pred = int(stats.mode(slice_preds, keepdims=False).mode)
    patient_true = int(stats.mode(test_lbls, keepdims=False).mode)

    print(f"Fold {fold_idx:3d}/{n_folds} | patient={patient_id} | "
          f"n_slices={len(test_lbls)} | slice_preds={np.bincount(slice_preds, minlength=NUM_CLASSES)} | "
          f"patient_pred={CLASS_NAMES[patient_pred]} | patient_true={CLASS_NAMES[patient_true]} | "
          f"{'CORRECT' if patient_pred == patient_true else 'WRONG'}")

    return patient_pred, patient_true


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

    REAL_PIDS_SOURCE = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_uint8/all_patient_ids.npy")
    real_pids_fixed = np.load(REAL_PIDS_SOURCE)
    real_test_candidates = sorted(np.unique(real_pids_fixed).tolist())
    n_folds = len(real_test_candidates)
    print(f"Loaded {len(all_images)} slices, {n_folds} patient folds\n")

    all_patient_preds, all_patient_true = [], []
    for fold_idx, pid in enumerate(real_test_candidates, start=1):
        result = run_fold(fold_idx, n_folds, pid, all_images, all_labels, all_pids)
        if result is not None:
            pred, true = result
            all_patient_preds.append(pred)
            all_patient_true.append(true)

    f1_per_class = f1_score(all_patient_true, all_patient_preds,
                            labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    macro_f1 = float(np.mean(f1_per_class))
    accuracy = float(np.mean(np.array(all_patient_preds) == np.array(all_patient_true)))

    print("\n" + "=" * 60)
    print(f"PATIENT-LEVEL LOPO Summary ({len(all_patient_preds)} patients) -- {args.data_dir}")
    print("=" * 60)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15} F1 = {f1_per_class[i]:.4f}")
    print(f"  {'Macro (5 classes)':<15} F1 = {macro_f1:.4f}")
    print(f"  {'Patient accuracy':<15} = {accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
