"""
data_helpers.py
---------------
Loads CT scan patches from .npy files and returns PyTorch DataLoaders.

Image preprocessing pipeline:
  1. Load raw int16 HU values (range typically -2443 to +2289).
  2. Min-max normalize to [0.0, 1.0] using global dataset min/max.
  3. Add channel dim: (N, H, W) → (N, 1, H, W)  [grayscale CT slice]
  4. Convert to float32 PyTorch tensors.

Labels:
  - Cast to int64 (required by CrossEntropyLoss).
  - Returned as plain Python int per sample to avoid NumPy view bugs.

Functions:
  get_data_loaders()   -- legacy fixed-split loader (kept for compatibility)
  get_lopo_loaders()   -- LOPO CV fold loader (preferred)
"""

import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class MyDataset(data.Dataset):
    """Simple dataset wrapper; stores pre-processed tensors."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # images: (N, 1, H, W), float32, values in [0, 1]
        # labels: (N,), int64
        self.images = images   # stays as numpy; converted in __getitem__
        self.labels = labels

    def __getitem__(self, index):
        # .copy() ensures a fresh contiguous memory block (avoids strided-view issues)
        img   = torch.from_numpy(self.images[index].copy())     # (1, H, W) float32
        label = int(self.labels[index])                          # plain Python int → LongTensor via DataLoader
        return img, label

    def __len__(self):
        return len(self.images)


def get_data_loaders(train_imgs_path, train_lbls_path, 
                     test_imgs_path, test_lbls_path, 
                     batch_size):
    """
    Build train / val DataLoaders from explicit .npy splits.
    Uses the Test set (LOPO) as the Validation set during training.
    """

    # ── 1. Load raw data ────────────────────────────────────────────────────
    train_images = np.load(train_imgs_path)                            # (N1, H, W)
    train_labels = np.load(train_lbls_path).astype(np.int64)           # (N1,)
    
    test_images  = np.load(test_imgs_path)                             # (N2, H, W)
    test_labels  = np.load(test_lbls_path).astype(np.int64)            # (N2,)

    # ── 2. Min-max normalization → [0.0, 1.0] ──────────────────────────────
    # Works for CT HU values 
    # Calculate global min/max ideally from train, but min/max across all is fine here
    img_min = min(float(train_images.min()), float(test_images.min()))
    img_max = max(float(train_images.max()), float(test_images.max()))
    
    train_images = (train_images.astype(np.float32) - img_min) / (img_max - img_min + 1e-8)
    test_images  = (test_images.astype(np.float32) - img_min) / (img_max - img_min + 1e-8)

    # ── 3. Add channel dimension: (N, H, W) → (N, 1, H, W) ─────────────────
    train_images = train_images[:, np.newaxis, :, :]                     
    test_images  = test_images[:, np.newaxis, :, :]                     

    # ── 4. DataLoaders ───────────────────────────────────────────────────────
    def make_loader(imgs, lbls, shuffle):
        ds = MyDataset(imgs, lbls)
        return data.DataLoader(ds, batch_size=batch_size,
                               shuffle=shuffle, num_workers=0, pin_memory=False)

    train_loader = make_loader(train_images, train_labels, shuffle=True)
    val_loader   = make_loader(test_images,  test_labels,  shuffle=False) # Use test set as val

    print(f"[data_helpers] train={len(train_images)}  val(test)={len(test_images)}")
    print(f"[data_helpers] image shape={train_images.shape[1:]}  "
          f"pixel range=[{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"[data_helpers] label classes: {np.unique(train_labels).tolist()}")

    # We return val_loader twice to match the unpack signature in classification_entry.py (train, val, test)
    # Since we are using test as val.
    return train_loader, val_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# LOPO CV loader
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(train_imgs: np.ndarray, test_imgs: np.ndarray):
    """Min-max normalize using global min/max across train + test."""
    img_min = min(float(train_imgs.min()), float(test_imgs.min()))
    img_max = max(float(train_imgs.max()), float(test_imgs.max()))
    denom   = img_max - img_min + 1e-8
    train_out = (train_imgs.astype(np.float32) - img_min) / denom
    test_out  = (test_imgs.astype(np.float32)  - img_min) / denom
    return train_out, test_out


def get_lopo_loaders(
    all_images:      np.ndarray,   # (N, H, W) int16
    all_labels:      np.ndarray,   # (N,)       int64
    all_patient_ids: np.ndarray,   # (N,)       int64; -1 = permanent train
    test_patient_id: int,
    batch_size:      int = 16,
):
    """
    Build train / test DataLoaders for one LOPO fold.

    Train = permanent patches (patient_id == -1)
            + all other LOPO patches (patient_id != test_patient_id AND >= 1)
    Test  = patches where patient_id == test_patient_id

    Returns: (train_loader, test_loader, n_train, n_test)
    """
    perm_mask  = all_patient_ids == -1
    test_mask  = all_patient_ids == test_patient_id
    train_mask = perm_mask | (~test_mask & (all_patient_ids >= 1)) | (all_patient_ids == -2)
    
    train_imgs = all_images[train_mask]
    train_lbls = all_labels[train_mask].astype(np.int64)
    test_imgs  = all_images[test_mask]
    test_lbls  = all_labels[test_mask].astype(np.int64)

    # Normalize
    train_imgs, test_imgs = _normalize(train_imgs, test_imgs)

    # Add channel dim: (N, H, W) → (N, 1, H, W)
    train_imgs = train_imgs[:, np.newaxis, :, :]
    test_imgs  = test_imgs[:,  np.newaxis, :, :]

    def make_loader(imgs, lbls, shuffle):
        ds = MyDataset(imgs, lbls)
        return data.DataLoader(ds, batch_size=batch_size,
                               shuffle=shuffle, num_workers=0, pin_memory=False)

    train_loader = make_loader(train_imgs, train_lbls, shuffle=True)
    test_loader  = make_loader(test_imgs,  test_lbls,  shuffle=False)

    return train_loader, test_loader, len(train_imgs), len(test_imgs)
