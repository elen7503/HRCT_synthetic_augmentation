"""
Leave-One-Patient-Out (LOPO) Cross-Validation for ILD patch classification.
"""

import os
import sys
import argparse

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.join(PROJECT_ROOT, "classifier_lib"))
sys.path.append(os.path.join(PROJECT_ROOT, "classifier_lib/Lung_Classification"))

from models import Classifier
from data_helpers import get_lopo_loaders
from train_utils import Logger, MetricTracker

ALL_IMGS_PATH = os.path.join(PROJECT_ROOT, "patches/ILD_DB_npy/all_images.npy")
ALL_LBLS_PATH = os.path.join(PROJECT_ROOT, "patches/ILD_DB_npy/all_labels.npy")
ALL_PIDS_PATH = os.path.join(PROJECT_ROOT, "patches/ILD_DB_npy/all_patient_ids.npy")

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

    net = Classifier(num_classes=NUM_CLASSES).to(DEVICE)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=train_lbls,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 1 if dry_run else NUM_EPOCHS

    for epoch in range(epochs):
        train_loss, train_tracker = train_one_epoch(net, train_loader, criterion, optimizer, DEVICE)
        test_loss,  test_tracker  = evaluate(net, test_loader, criterion, DEVICE)
        scheduler.step()

        train_acc = (np.array(train_tracker.y_true) == np.array(train_tracker.y_pred)).mean() * 100
        test_acc  = (np.array(test_tracker.y_true)  == np.array(test_tracker.y_pred)).mean()  * 100

        logger.log(f"  Epoch {epoch+1:02d}/{epochs}"
                   f"  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}"
                   f"  train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%")
     
    final_epoch = epochs
    final_test_acc = test_acc
    final_ckpt_path = os.path.join(logger.ckpt_dir, f"fold_{fold_idx:03d}_pid_{patient_id}_final.pth")
    torch.save(
        {
            "fold_idx": fold_idx,
            "patient_id": patient_id,
            "final_epoch": final_epoch,
            "final_test_acc": final_test_acc,
            "state_dict": net.state_dict(),
        },
        final_ckpt_path,
    )
    best_tracker = test_tracker

    f1_per_class = f1_score(best_tracker.y_true, best_tracker.y_pred,
                            labels=list(range(NUM_CLASSES)),
                            average=None, zero_division=0)
    macro_f1_all_classes = float(np.mean(f1_per_class))

    present_classes = sorted(set(best_tracker.y_true))
    macro_f1_present_classes = f1_score(
        best_tracker.y_true,
        best_tracker.y_pred,
        labels=present_classes,
        average="macro",
        zero_division=0,
    )

    logger.log(f"  Final-epoch test acc: {final_test_acc:.2f}%")
    logger.log(f"  Final epoch: {final_epoch:02d}  |  ckpt: {final_ckpt_path}")
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
        "final_epoch": final_epoch,
        "final_test_acc": final_test_acc,
        "final_ckpt_path": final_ckpt_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only the first fold with 1 epoch (smoke test).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for this run (weight init, batch order)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Using device: {DEVICE}")
    print("Loading data...")
    all_images = np.load(ALL_IMGS_PATH)
    all_labels = np.load(ALL_LBLS_PATH).astype(np.int64)
    all_pids   = np.load(ALL_PIDS_PATH).astype(np.int64)
    print(f"  Total patches: {len(all_images)}")

    lopo_patient_ids = sorted(np.unique(all_pids[all_pids >= 1]).tolist())
    n_folds = len(lopo_patient_ids)
    print(f"  LOPO patients: {n_folds}")

    if args.dry_run:
        lopo_patient_ids = lopo_patient_ids[:1]
        print("  [DRY RUN] Running only fold 1 with 1 epoch.\n")

    logger = Logger(f"lopo_baseline_v4_seed{args.seed}")
    logger.log(f"LOPO CV | device={DEVICE} | epochs={1 if args.dry_run else NUM_EPOCHS}"
               f" | lr={LR} | batch={BATCH_SIZE}")
    logger.log(f"Total folds: {n_folds}  |  classes: {CLASS_NAMES}")

    all_f1s = []
    all_macro_f1_all = []
    all_macro_f1_present = []

    for fold_idx, patient_id in enumerate(lopo_patient_ids, start=1):
        metrics = run_fold(fold_idx, n_folds, patient_id,
                           all_images, all_labels, all_pids,
                           logger, dry_run=args.dry_run)
        if metrics is not None:
            all_f1s.append(metrics["f1_per_class"])
            all_macro_f1_all.append(metrics["macro_f1_all_classes"])
            all_macro_f1_present.append(metrics["macro_f1_present_classes"])

    if all_f1s:
        all_f1s = np.array(all_f1s)
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

        summary_path = os.path.join(logger.exp_dir, "lopo_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"\nSummary saved to: {summary_path}")

    logger.close()
