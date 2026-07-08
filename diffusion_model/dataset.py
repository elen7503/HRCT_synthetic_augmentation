"""
dataset.py
----------
DataLoader for a single tissue class from the Talisman Test Suite .npy files.
Normalises HU values to [-1, 1] for DDPM training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = {
    0: 'healthy',
    1: 'emphysema',
    2: 'ground_glass',
    3: 'fibrosis',
    4: 'micronodules',
}

# HU clip range — covers relevant lung tissue values
HU_MIN = -1000
HU_MAX =  500


def normalise(x):
    """Clip HU values and normalise to [-1, 1]."""
    x = np.clip(x, HU_MIN, HU_MAX).astype(np.float32)
    x = (x - HU_MIN) / (HU_MAX - HU_MIN)  # [0, 1]
    x = x * 2.0 - 1.0                       # [-1, 1]
    return x


class PatchDataset(Dataset):
    """
    Loads all patches of a single class from the unified .npy arrays.
    Uses all available patches (LOPO + permanent train) for generative training,
    since we are not evaluating the diffusion model via LOPO.
    """
    def __init__(self, npy_dir, class_idx):
        images = np.load(f"{npy_dir}/all_images.npy")
        labels = np.load(f"{npy_dir}/all_labels.npy")

        mask = labels == class_idx
        self.patches = normalise(images[mask])
        self.class_idx = class_idx
        self.class_name = CLASS_NAMES[class_idx]

        print(f"Class '{self.class_name}': {len(self.patches)} patches loaded.")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = self.patches[idx]                        # (32, 32)
        x = torch.tensor(x).unsqueeze(0)            # (1, 32, 32)
        return x


def get_dataloader(npy_dir, class_idx, batch_size=64, num_workers=2):
    dataset = PatchDataset(npy_dir, class_idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
