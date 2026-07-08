import os
import sys
import re
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Add parent directory to path to import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Lung_Segmentation")))

from models import U_Net, AttU_Net, R2U_Net, R2AttU_Net
from train_utils import Logger, MetricTracker

# ── Config ──────────────────────────────────────────────────────────────────
# Choose from: 'ILD_DB_lungMasks', 'ILD_DB_txtROIs', 'ILD_DB_volumeROIs'
DATASET_NAME = "ILD_DB_volumeROIs" 
DATA_ROOT    = os.path.join("ILD_DB_fig", DATASET_NAME)

IMG_SIZE     = 256
BATCH_SIZE   = 4
NUM_EPOCHS   = 20
LR           = 1e-4
TRAIN_SPLIT  = 0.8
DEVICE       = torch.device("cpu") # Default to CPU for stability

# NEW: Filter out slices where the mask is entirely empty (black)
# - For lungMasks, empty means no lung found (good to filter)
# - For volumeROIs, empty means Healthy lung (MUST KEEP)
FILTER_EMPTY_MASKS = (DATASET_NAME == "ILD_DB_lungMasks")

if DATASET_NAME == "ILD_DB_lungMasks":
    NUM_CLASSES = 2
    CLASS_NAMES = ["Background", "Lung"]
    MASK_SUBDIR = "lung_mask"
else:
    # According to ILD_DB README, labels go from 1 to 17 (plus 0 for bg) = 18 classes
    NUM_CLASSES  = 18         # 0=background, 1–17 disease classes
    CLASS_NAMES = [
        "Empty/Background", "Healthy", "Emphysema", "Ground Glass", "Fibrosis",
        "Micronodules", "Consolidation", "Bronchial Wall Thick", "Reticulation", 
        "Macronodules", "Cysts", "Peripheral Micronodules", "Bronchiectasis",
        "Air Trapping", "Early Fibrosis", "Increased Attenuation", "Tuberculosis", "PCP"
    ]
    MASK_SUBDIR = "roi_mask"

# ── Dataset ──────────────────────────────────────────────────────────────────
def _extract_slice_num(path):
    # Matches -0001.dcm.jpg, _0001.dcm.jpg, 0001.jpg, 0001.png, etc.
    m = re.search(r'[-_](\d+)(\.dcm)?\.(jpg|png)$', os.path.basename(path))
    return int(m.group(1)) if m else -1

def build_pairs(data_root, mask_subdir, filter_empty=False):
    pairs = []
    if not os.path.exists(data_root):
        print(f"Error: {data_root} Does not exist.")
        return pairs
        
    for patient in sorted(os.listdir(data_root)):
        patient_dir = os.path.join(data_root, patient)
        if not os.path.isdir(patient_dir): continue
        
        mask_dirs = []
        for root, dirs, files in os.walk(patient_dir):
            if mask_subdir in dirs:
                mask_dirs.append(os.path.join(root, mask_subdir))
        
        if not mask_dirs:
            continue

        for mask_dir in mask_dirs:
            ct_dir = os.path.dirname(mask_dir)
            ct_files = {
                _extract_slice_num(f): f
                for f in glob.glob(os.path.join(ct_dir, "*.dcm.jpg"))
            }

            # Masks might be .jpg (old corrupted lossy format) or .png (new lossless format)
            mask_files = glob.glob(os.path.join(mask_dir, "*.jpg")) + glob.glob(os.path.join(mask_dir, "*.png"))
            for mask_path in mask_files:
                snum = _extract_slice_num(mask_path)
                if snum in ct_files:
                    # Optional filtering of empty masks
                    if filter_empty:
                        m_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if m_img is None or np.sum(m_img) == 0:
                            continue
                    pairs.append((ct_files[snum], mask_path))
    return pairs

class SegDataset(Dataset):
    def __init__(self, pairs, img_size=256, num_classes=2):
        self.pairs       = pairs
        self.img_size    = img_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ct_path, mask_path = self.pairs[idx]

        # Load CT slice (grayscale)
        ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
        if ct is None:
            # Defense against corrupted files
            ct = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            ct = cv2.resize(ct, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            
        ct = ct.astype(np.float32) / 255.0
        ct = torch.from_numpy(ct).unsqueeze(0)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
        # Robust handling of binary masks (e.g. 255 -> 1)
        if self.num_classes == 2 and mask.max() > 1:
            mask = (mask > 0).astype(np.int64)
            
        mask = mask.astype(np.int64)
        mask = np.clip(mask, 0, self.num_classes - 1)
        mask = torch.from_numpy(mask)

        return ct, mask

# ── Setup ──────────────────────────────────────────────────────────────────
logger = Logger(f"segmentation_{DATASET_NAME}")
logger.log(f"Using device: {DEVICE}")

logger.log(f"Scanning dataset: {DATASET_NAME} (Filter empty masks: {FILTER_EMPTY_MASKS}) ...")
pairs = build_pairs(DATA_ROOT, MASK_SUBDIR, filter_empty=FILTER_EMPTY_MASKS)
logger.log(f"Found {len(pairs)} non-empty CT-mask pairs.")

if len(pairs) == 0:
    logger.log("ERROR: No pairs found.")
    sys.exit(1)

train_pairs, val_pairs = train_test_split(pairs, test_size=1 - TRAIN_SPLIT, random_state=42)
logger.log(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}\n")

train_loader = DataLoader(SegDataset(train_pairs, IMG_SIZE, NUM_CLASSES), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SegDataset(val_pairs,   IMG_SIZE, NUM_CLASSES), batch_size=BATCH_SIZE, shuffle=False)

# Choose model: AttU_Net
net       = AttU_Net(img_ch=1, output_ch=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

best_val_loss = float("inf")

# ── Training Loop ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    for epoch in range(NUM_EPOCHS):
        # ---- Train ----
        net.train()
        train_loss = 0.0
        tracker = MetricTracker(NUM_CLASSES, task_type='segmentation')

        for ct, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            ct   = ct.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()
            pred = net(ct)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            tracker.update(mask, pred)

        train_loss /= len(train_loader)
        
        # ---- Validation ----
        net.eval()
        val_loss = 0.0
        val_tracker = MetricTracker(NUM_CLASSES, task_type='segmentation')

        with torch.no_grad():
            for ct, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]  "):
                ct   = ct.to(DEVICE)
                mask = mask.to(DEVICE)
                pred = net(ct)
                loss = criterion(pred, mask)
                val_loss += loss.item()
                val_tracker.update(mask, pred)

        val_loss /= len(val_loader)
        
        # Report metrics
        logger.log(f"\n[Epoch {epoch+1:02d}/{NUM_EPOCHS}]")
        logger.log(f"Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
        logger.log("--- Validation Report ---")
        logger.log(val_tracker.get_report(class_names=CLASS_NAMES))
        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.log(f"  ↳ New Best Val Loss: {val_loss:.4f}")
        
        logger.save_ckpt(net, optimizer, epoch + 1, is_best=is_best)
        scheduler.step()

    logger.log("\nDone!")
    logger.close()
