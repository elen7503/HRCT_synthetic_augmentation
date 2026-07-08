"""
classification_entry.py
-----------------------
Leave-One-Patient-Out (LOPO) Cross-Validation for ILD patch classification.

For each patient fold:
  Train = permanent patches (patient_id == -1)
          + LOPO patches from all other patients
  Test  = LOPO patches from the held-out patient

Runs NUM_EPOCHS of training per fold, then aggregates results.
Final summary written to experiments/lopo_<timestamp>/lopo_summary.txt
"""

import os
import sys
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

# Add parent directory to path to import models and data_helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Lung_Classification")))

from models import Classifier
from data_helpers import get_lopo_loaders
from train_utils import Logger, MetricTracker

# ── Config ───────────────────────────────────────────────────────────────────
ALL_IMGS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/all_images.npy"
ALL_LBLS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/all_labels.npy"
ALL_PIDS_PATH = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/all_patient_ids.npy"

BATCH_SIZE  = 16
NUM_EPOCHS  = 15
NUM_CLASSES = 5
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Healthy", "Emphysema", "Ground Glass", "Fibrosis", "Micronodules"]


def train_one_epoch(net, loader, criterion, optimizer, device):
    net.train()
    total_loss = 0.0
    tracker = MetricTracker(NUM_CLASSES, task_type='classification')
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        tracker.update(labels, predicted)
    return total_loss / len(loader), tracker


def evaluate(net, loader, criterion, device):
    net.eval()
    total_loss = 0.0
    tracker = MetricTracker(NUM_CLASSES, task_type='classification')
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            tracker.update(labels, predicted)
    return total_loss / len(loader), tracker


def run_fold(fold_idx, n_folds, patient_id, all_images, all_labels, all_pids,
             logger, dry_run=False):
    """Train + evaluate one LOPO fold. Returns metrics dict."""

    train_loader, test_loader, n_train, n_test = get_lopo_loaders(
        all_images, all_labels, all_pids,
        test_patient_id=patient_id,
        batch_size=BATCH_SIZE,
    )

    logger.log(f"\n{'='*60}")
    logger.log(f"Fold {fold_idx:3d}/{n_folds}  |  patient_id={patient_id}"
               f"  |  train={n_train}  test={n_test}")

    if n_test == 0:
        logger.log("  [SKIP] No test patches for this patient.")
        return None

    # Warn if any class is missing from train
    train_lbls = all_labels[(all_pids == -1) | ((all_pids >= 1) & (all_pids != patient_id))]
    missing = [CLASS_NAMES[c] for c in range(NUM_CLASSES) if not (train_lbls == c).any()]
    if missing:
        logger.log(f"  [WARN] Classes missing from train: {missing}")

    # Fresh model per fold
    net = Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 1 if dry_run else NUM_EPOCHS
    best_test_acc = 0.0
    best_epoch = 1
    best_state_dict = copy.deepcopy(net.state_dict())

    for epoch in range(epochs):
        train_loss, train_tracker = train_one_epoch(net, train_loader, criterion, optimizer, DEVICE)
        test_loss,  test_tracker  = evaluate(net, test_loader, criterion, DEVICE)
        scheduler.step()

        train_acc = (np.array(train_tracker.y_true) == np.array(train_tracker.y_pred)).mean() * 100
        test_acc  = (np.array(test_tracker.y_true)  == np.array(test_tracker.y_pred)).mean()  * 100

        logger.log(f"  Epoch {epoch+1:02d}/{epochs}"
                   f"  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}"
                   f"  train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(net.state_dict())

    # Save and reload best model for final test metrics
    best_ckpt_path = os.path.join(logger.ckpt_dir, f"fold_{fold_idx:03d}_pid_{patient_id}_best.pth")
    torch.save(
        {
            "fold_idx": fold_idx,
            "patient_id": patient_id,
            "best_epoch": best_epoch,
            "best_test_acc": best_test_acc,
            "state_dict": best_state_dict,
        },
        best_ckpt_path,
    )
    net.load_state_dict(best_state_dict)
    _, best_tracker = evaluate(net, test_loader, criterion, DEVICE)

    # Final per-class F1 on the test set (5 classes fixed)
    f1_per_class = f1_score(best_tracker.y_true, best_tracker.y_pred,
                            labels=list(range(NUM_CLASSES)),
                            average=None, zero_division=0)
    macro_f1_all_classes = float(np.mean(f1_per_class))

    # Macro-F1 only over classes that appear in this fold's test set
    present_classes = sorted(set(best_tracker.y_true))
    macro_f1_present_classes = f1_score(
        best_tracker.y_true,
        best_tracker.y_pred,
        labels=present_classes,
        average="macro",
        zero_division=0,
    )

    logger.log(f"  Best test acc: {best_test_acc:.2f}%")
    logger.log(f"  Best epoch: {best_epoch:02d}  |  best ckpt: {best_ckpt_path}")
    logger.log(f"  Per-class F1: " +
               " | ".join(f"{CLASS_NAMES[i]}={f1_per_class[i]:.3f}" for i in range(NUM_CLASSES)))
    logger.log(f"  Macro-F1 (5 fixed classes): {macro_f1_all_classes:.3f}")
    logger.log(f"  Macro-F1 (present test classes only): {macro_f1_present_classes:.3f}"
               f"  |  present={present_classes}")

    return {
        "f1_per_class": f1_per_class,
        "macro_f1_all_classes": macro_f1_all_classes,
        "macro_f1_present_classes": macro_f1_present_classes,
        "present_classes": present_classes,
        "best_epoch": best_epoch,
        "best_test_acc": best_test_acc,
        "best_ckpt_path": best_ckpt_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only the first fold with 1 epoch (smoke test).")
    args = parser.parse_args()

    # ── Load all data into memory (fast indexing per fold) ───────────────────
    print(f"Using device: {DEVICE}")
    print("Loading data...")
    all_images = np.load(ALL_IMGS_PATH)
    all_labels = np.load(ALL_LBLS_PATH).astype(np.int64)
    all_pids   = np.load(ALL_PIDS_PATH).astype(np.int64)
    print(f"  Total patches: {len(all_images)}")

    # LOPO patient IDs (exclude -1 permanent train patches)
    lopo_patient_ids = sorted(np.unique(all_pids[all_pids >= 1]).tolist())
    n_folds = len(lopo_patient_ids)
    print(f"  LOPO patients: {n_folds}")

    if args.dry_run:
        lopo_patient_ids = lopo_patient_ids[:1]
        print("  [DRY RUN] Running only fold 1 with 1 epoch.\n")

    # ── Logger ───────────────────────────────────────────────────────────────
    logger = Logger("lopo_classification")
    logger.log(f"LOPO CV | device={DEVICE} | epochs={1 if args.dry_run else NUM_EPOCHS}"
               f" | lr={LR} | batch={BATCH_SIZE}")
    logger.log(f"Total folds: {n_folds}  |  classes: {CLASS_NAMES}")

    # ── LOPO loop ────────────────────────────────────────────────────────────
    all_f1s = []                    # list of (NUM_CLASSES,) arrays
    all_macro_f1_all = []           # list of floats (5 fixed classes)
    all_macro_f1_present = []       # list of floats (present classes only)

    for fold_idx, patient_id in enumerate(lopo_patient_ids, start=1):
        metrics = run_fold(fold_idx, n_folds, patient_id,
                           all_images, all_labels, all_pids,
                           logger, dry_run=args.dry_run)
        if metrics is not None:
            all_f1s.append(metrics["f1_per_class"])
            all_macro_f1_all.append(metrics["macro_f1_all_classes"])
            all_macro_f1_present.append(metrics["macro_f1_present_classes"])

    # ── Aggregate summary ────────────────────────────────────────────────────
    if all_f1s:
        all_f1s = np.array(all_f1s)   # (n_valid_folds, NUM_CLASSES)
        mean_f1 = all_f1s.mean(axis=0)
        std_f1  = all_f1s.std(axis=0)

        summary = "\n" + "="*60
        summary += f"\nLOPO CV Summary  ({len(all_f1s)} folds)\n"
        summary += "="*60 + "\n"
        for i, name in enumerate(CLASS_NAMES):
            summary += f"  {name:<18} F1 = {mean_f1[i]:.4f} ± {std_f1[i]:.4f}\n"
        summary += f"  {'Macro (5 classes)':<18} F1 = {np.mean(all_macro_f1_all):.4f} ± {np.std(all_macro_f1_all):.4f}\n"
        summary += f"  {'Macro (present only)':<18} F1 = {np.mean(all_macro_f1_present):.4f} ± {np.std(all_macro_f1_present):.4f}\n"
        summary += "="*60

        logger.log(summary)

        # Save summary to standalone file
        summary_path = os.path.join(logger.exp_dir, "lopo_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"\nSummary saved to: {summary_path}")

    logger.close()
