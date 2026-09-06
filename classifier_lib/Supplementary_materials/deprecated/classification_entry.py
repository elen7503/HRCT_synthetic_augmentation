import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add parent directory to path to import models and data_helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Lung_Classification")))

from models import Classifier
from data_helpers import get_data_loaders
from train_utils import Logger, MetricTracker

# ── Config ──────────────────────────────────────────────────────────────────
TRAIN_IMGS_PATH = "ILD_DB_npy/train_images.npy"
TRAIN_LBLS_PATH = "ILD_DB_npy/train_labels.npy"
TEST_IMGS_PATH  = "ILD_DB_npy/test_images.npy"
TEST_LBLS_PATH  = "ILD_DB_npy/test_labels.npy"

BATCH_SIZE   = 16
NUM_EPOCHS   = 15
NUM_CLASSES  = 5
IMG_DIM      = 32
LR           = 1e-3
DEVICE       = torch.device("cpu") 

# Mapping for 5 classes
CLASS_NAMES = ["Healthy", "Emphysema", "Ground Glass", "Fibrosis", "Micronodules"]

# ── Setup ──────────────────────────────────────────────────────────────────
logger = Logger("classification")
logger.log(f"Using device: {DEVICE}")

# Using explicit train and test splits to avoid data leakage
train_loader, val_loader, test_loader = get_data_loaders(
    train_imgs_path=TRAIN_IMGS_PATH,
    train_lbls_path=TRAIN_LBLS_PATH,
    test_imgs_path=TEST_IMGS_PATH,
    test_lbls_path=TEST_LBLS_PATH,
    batch_size=BATCH_SIZE
)

net = Classifier(num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

best_val_acc = 0.0

# ── Training Loop ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        # ---- Train ----
        net.train()
        train_loss = 0.0
        tracker = MetricTracker(NUM_CLASSES, task_type='classification')

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            tracker.update(labels, predicted)

        train_loss /= len(train_loader)
        
        # ---- Validation ----
        net.eval()
        val_loss = 0.0
        val_tracker = MetricTracker(NUM_CLASSES, task_type='classification')

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  "):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_tracker.update(labels, predicted)

        val_loss /= len(val_loader)
        
        # Report metrics
        logger.log(f"\n[Epoch {epoch+1:02d}/{NUM_EPOCHS}]")
        logger.log(f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        logger.log("--- Validation Report ---")
        logger.log(val_tracker.get_report(class_names=CLASS_NAMES))
        
        # Checkpoint
        val_acc = (np.array(val_tracker.y_true) == np.array(val_tracker.y_pred)).mean() * 100
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            logger.log(f"  ↳ New Best Val Acc: {val_acc:.2f}%")
        
        logger.save_ckpt(net, optimizer, epoch + 1, is_best=is_best)
        scheduler.step()

    # ── Final Test ─────────────────────────────────────────────────────────────
    logger.log("\n" + "="*30)
    logger.log("Evaluating on test set...")
    net.eval()
    test_tracker = MetricTracker(NUM_CLASSES, task_type='classification')

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            test_tracker.update(labels, predicted)

    logger.log("--- Final Test Report ---")
    logger.log(test_tracker.get_report(class_names=CLASS_NAMES))
    logger.close()
