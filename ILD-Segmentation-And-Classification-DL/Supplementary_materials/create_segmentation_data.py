"""
segmentation_entry.py
----------------------
Entry script for training the ILD lung segmentation models (UNet, AttUNet, etc.)
using the ILD_DB_fig dataset.

Data layout (ILD_DB_fig):
  ILD_DB_volumeROIs/
    <patient_id>/
      CT-XXXX-YYYY.dcm.jpg   ← CT slices  (512×512 grayscale)
      roi_mask/
        roi_mask_XXXX_YY.dcm.jpg  ← disease segmentation masks (512×512)
                                      pixel values = label index (1–12 used)
                                      0 = background / unlabelled

  ILD_DB_lungMasks/
    <patient_id>/
      CT-XXXX-YYYY.dcm.jpg   ← CT slices (same as above)
      lung_mask/
        lung_mask_XXXX_YY.dcm.jpg ← binary lung masks (0=bg, 255=lung)
"""

import os, sys, re, glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.abspath("Lung_Segmentation"))
from models import U_Net, AttU_Net, R2U_Net, R2AttU_Net

# ── Config ──────────────────────────────────────────────────────────────────
DATA_ROOT    = "ILD_DB_fig/ILD_DB_volumeROIs"   # paired CT + roi_mask
IMG_SIZE     = 256        # resize to reduce memory (original 512x512, use 256 for speed)
NUM_CLASSES  = 13         # 0=background, 1–12 disease classes
BATCH_SIZE   = 4
NUM_EPOCHS   = 20
LR           = 1e-4
TRAIN_SPLIT  = 0.8
# DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE       = torch.device("cpu")

print(f"Using device: {DEVICE}")

# ── Dataset ──────────────────────────────────────────────────────────────────
def _extract_slice_num(path):
    """Pull the numeric slice index from a filename for matching CT <-> mask."""
    m = re.search(r'[-_](\d+)\.dcm\.jpg$', os.path.basename(path))
    return int(m.group(1)) if m else -1

def build_pairs(data_root):
    """Walk `data_root` and collect (ct_path, mask_path) pairs."""
    pairs = []
    for patient in sorted(os.listdir(data_root)):
        patient_dir  = os.path.join(data_root, patient)
        mask_dir     = os.path.join(patient_dir, "roi_mask")
        if not os.path.isdir(mask_dir):
            continue

        # Index CT slices by slice number
        ct_files = {
            _extract_slice_num(f): f
            for f in glob.glob(os.path.join(patient_dir, "*.dcm.jpg"))
        }

        for mask_path in glob.glob(os.path.join(mask_dir, "*.jpg")):
            snum = _extract_slice_num(mask_path)
            if snum in ct_files:
                pairs.append((ct_files[snum], mask_path))

    return pairs


class SegDataset(Dataset):
    def __init__(self, pairs, img_size=256):
        self.pairs    = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ct_path, mask_path = self.pairs[idx]

        # Load CT slice (grayscale)
        ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
        ct = cv2.resize(ct, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        ct = ct.astype(np.float32) / 255.0           # normalize to [0, 1]
        ct = torch.from_numpy(ct).unsqueeze(0)        # (1, H, W)

        # Load mask (label-encoded: pixel value = class 0–12)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int64)
        # Clamp anything above num_classes to 0 (background)  
        mask = np.clip(mask, 0, NUM_CLASSES - 1)
        mask = torch.from_numpy(mask)                 # (H, W) long

        return ct, mask


# ── Build Pairs & DataLoaders ────────────────────────────────────────────────
print("Scanning dataset...")
pairs = build_pairs(DATA_ROOT)
print(f"Found {len(pairs)} CT-mask pairs across all patients.")

if len(pairs) == 0:
    print("ERROR: No pairs found. Check that DATA_ROOT is correct.")
    sys.exit(1)

train_pairs, val_pairs = train_test_split(pairs, test_size=1 - TRAIN_SPLIT, random_state=42)
print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

train_loader = DataLoader(SegDataset(train_pairs, IMG_SIZE), batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(SegDataset(val_pairs,   IMG_SIZE), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

# ── Model, Loss, Optimizer ───────────────────────────────────────────────────
# Choose model: U_Net | AttU_Net | R2U_Net | R2AttU_Net
net       = AttU_Net(img_ch=1, output_ch=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()   # expects (B, C, H, W) logits and (B, H, W) labels
optimizer = optim.Adam(net.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

print(f"Model: AttU_Net  |  Classes: {NUM_CLASSES}  |  Input: {IMG_SIZE}×{IMG_SIZE}")
print("Starting training...\n")

best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    # ── Train ──────────────────────────────────────────────────────────────
    net.train()
    train_loss = 0.0

    for ct, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
        ct   = ct.to(DEVICE)
        mask = mask.to(DEVICE)        # (B, H, W) long

        optimizer.zero_grad()
        pred = net(ct)                # (B, NUM_CLASSES, H, W)
        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── Validate ────────────────────────────────────────────────────────────
    net.eval()
    val_loss = 0.0

    with torch.no_grad():
        for ct, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  "):
            ct   = ct.to(DEVICE)
            mask = mask.to(DEVICE)
            pred = net(ct)
            loss = criterion(pred, mask)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"[Epoch {epoch+1:02d}/{NUM_EPOCHS}]  "
          f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(net.state_dict(), "best_seg_model.pth")
        print("  ↳ Saved best model.")

    scheduler.step()

print(f"\nDone! Best val loss: {best_val_loss:.4f}")
print("Model saved to: best_seg_model.pth")
