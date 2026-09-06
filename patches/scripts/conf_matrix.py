"""
build_confusion_matrix_from_checkpoints.py
--------------------------------------------
Reconstructs a pooled confusion matrix across all LOPO folds using the
best-checkpoint .pth files your classification_lopo.py run already saved,
WITHOUT retraining anything.

For each fold checkpoint:
  1. Parse patient_id from the filename.
  2. Rebuild that fold's test_loader (via the same get_lopo_loaders you
     already use).
  3. Load the saved best state_dict into a fresh Classifier.
  4. Run one evaluation pass, collect y_true/y_pred.

At the end: pools everything into one confusion matrix + per-class
present-fold counts (how many folds actually contained each class in
their test set), which is the piece of information your current
lopo_summary.txt does not give you.
"""

import os
import re
import glob
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report

import sys
sys.path.append("/rds/general/user/eh1121/home/Final_Project/classifier_lib")
sys.path.append("/rds/general/user/eh1121/home/Final_Project/classifier_lib/Lung_Classification")

from models import Classifier
from data_helpers import get_lopo_loaders

# ============================================================
# CONFIG -- edit these two paths
# ============================================================

ALL_IMGS_PATH = "/rds/general/user/eh1121/home/Final_Project/patches/ILD_DB_npy/all_images.npy"
ALL_LBLS_PATH = "/rds/general/user/eh1121/home/Final_Project/patches/ILD_DB_npy/all_labels.npy"
ALL_PIDS_PATH = "/rds/general/user/eh1121/home/Final_Project/patches/ILD_DB_npy/all_patient_ids.npy"

# Point this at the experiment directory your LOPO run created, e.g.
# "experiments/lopo_classification_20260706_143000"
EXPERIMENT_DIR = "/rds/general/user/eh1121/home/Final_Project/patches/experiments/lopo_classification_20260629_192500/"

NUM_CLASSES = 5
BATCH_SIZE = 16
CLASS_NAMES = ["Healthy", "Emphysema", "Ground Glass", "Fibrosis", "Micronodules"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================

def main():
    ckpt_dir = os.path.join(EXPERIMENT_DIR, "checkpoints")
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "fold_*_pid_*_best.pth")))
    if not ckpt_files:
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir}. Check EXPERIMENT_DIR."
        )
    print(f"Found {len(ckpt_files)} fold checkpoints.")

    all_images = np.load(ALL_IMGS_PATH)
    all_labels = np.load(ALL_LBLS_PATH).astype(np.int64)
    all_pids = np.load(ALL_PIDS_PATH).astype(np.int64)

    all_y_true, all_y_pred = [], []
    # how many folds actually had each class present in the test set at all
    class_present_fold_count = np.zeros(NUM_CLASSES, dtype=int)
    # per-class F1 computed ONLY over folds where that class was present
    per_class_f1_when_present = {c: [] for c in range(NUM_CLASSES)}

    pattern = re.compile(r"fold_(\d+)_pid_(-?\d+)_best\.pth")

    for ckpt_path in ckpt_files:
        m = pattern.search(os.path.basename(ckpt_path))
        if not m:
            print(f"  [SKIP] Could not parse patient_id from {ckpt_path}")
            continue
        patient_id = int(m.group(2))

        _, test_loader, _, n_test = get_lopo_loaders(
            all_images, all_labels, all_pids,
            test_patient_id=patient_id,
            batch_size=BATCH_SIZE,
        )
        if n_test == 0:
            continue

        net = Classifier(num_classes=NUM_CLASSES).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()

        fold_y_true, fold_y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                fold_y_true.extend(labels.cpu().numpy().tolist())
                fold_y_pred.extend(predicted.cpu().numpy().tolist())

        present = sorted(set(fold_y_true))
        for c in present:
            class_present_fold_count[c] += 1
            from sklearn.metrics import f1_score as _f1
            f1_c = _f1(fold_y_true, fold_y_pred, labels=[c], average=None, zero_division=0)[0]
            per_class_f1_when_present[c].append(f1_c)

        all_y_true.extend(fold_y_true)
        all_y_pred.extend(fold_y_pred)

    print(f"\nTotal test patches pooled across folds: {len(all_y_true)}")

    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(NUM_CLASSES)))
    print("\nPooled confusion matrix (rows=true, cols=predicted):")
    header = "        " + "".join(f"{n[:6]:>8}" for n in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i][:6]:>8}" + "".join(f"{v:>8}" for v in row))

    print("\nFull classification report (pooled across all folds' test sets):")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=CLASS_NAMES, digits=4, zero_division=0))

    print("\nHow many of the", len(ckpt_files), "folds actually had each class present in test:")
    for c, name in enumerate(CLASS_NAMES):
        n_present = class_present_fold_count[c]
        mean_f1_when_present = np.mean(per_class_f1_when_present[c]) if per_class_f1_when_present[c] else float("nan")
        print(f"  {name:<15} present in {n_present:3d} folds  |  "
              f"mean F1 IN THOSE FOLDS ONLY = {mean_f1_when_present:.4f}")

    print("\n--> Compare 'mean F1 when present' above to the lopo_summary.txt numbers.")
    print("    If a class's summary F1 was near 0 but its 'when present' F1 here is")
    print("    much higher, the low summary number was mostly an artifact of being")
    print("    averaged over folds where the class never appeared in test at all --")
    print("    not evidence the classifier fails on that class.")
    print("    If the two numbers roughly agree, the failure is real.")

    out_path = os.path.join(EXPERIMENT_DIR, "pooled_confusion_matrix.npz")
    np.savez(out_path, confusion_matrix=cm,
             y_true=np.array(all_y_true), y_pred=np.array(all_y_pred),
             class_present_fold_count=class_present_fold_count)
    print(f"\nSaved raw pooled predictions + confusion matrix to {out_path}")


if __name__ == "__main__":
    main()