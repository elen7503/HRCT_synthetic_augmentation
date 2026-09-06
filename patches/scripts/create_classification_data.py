"""
Reads all .tif patches from ILD_DB_talismanTestSuite and saves unified
numpy arrays that preserve patient ID, enabling LOPO cross-validation.
"""

import os
import glob
import re
import numpy as np
import cv2
from tqdm import tqdm

PROJECT_ROOT = os.environ["PROJECT_ROOT"]


CLASS_MAPPING = {
    'healthy':      0,
    'emphysema':    1,
    'ground_glass': 2,
    'fibrosis':     3,
    'micronodules': 4,
}


def parse_filename(filename):
    base = os.path.splitext(filename)[0]

    is_perm_train = 'patient-1_' in base

    match_class = re.match(r'^(.+?)_patch\d+_', base)
    if not match_class:
        return None, None
    class_name = match_class.group(1)

    # Verify known class
    if class_name not in CLASS_MAPPING:
        return None, None

    patient_id = -1 if is_perm_train else None

    if not is_perm_train:
        match_pid = re.search(r'_patient(\d+)', base)
        if match_pid:
            patient_id = int(match_pid.group(1))
        else:
            return None, None

    return class_name, patient_id


def prepare_lopo_dataset(data_dir, output_dir):
    image_paths = sorted(glob.glob(os.path.join(data_dir, '*.tif')))
    if not image_paths:
        print(f"No .tif files found in {data_dir}.")
        return

    print(f"Found {len(image_paths)} .tif files. Parsing...")

    images, labels, patient_ids = [], [], []
    skipped = 0

    for img_path in tqdm(image_paths, desc="Loading patches"):
        filename = os.path.basename(img_path)
        class_name, patient_id = parse_filename(filename)

        if class_name is None or patient_id is None:
            print(f"  [SKIP] Cannot parse: {filename}")
            skipped += 1
            continue

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [SKIP] Failed to read: {filename}")
            skipped += 1
            continue

        if img.shape[:2] != (32, 32):
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_NEAREST)

        images.append(img)
        labels.append(CLASS_MAPPING[class_name])
        patient_ids.append(patient_id)

    imgs_np = np.array(images,      dtype=np.int16)
    lbls_np = np.array(labels,      dtype=np.int64)
    pids_np = np.array(patient_ids, dtype=np.int64)

    print(f"\nTotal loaded : {len(imgs_np)}  |  Skipped: {skipped}")
    print("\n--- Permanent train patches (patient_id == -1) ---")
    perm_mask = pids_np == -1
    perm_lbls = lbls_np[perm_mask]
    for name, idx in CLASS_MAPPING.items():
        print(f"  {name}: {int((perm_lbls == idx).sum())}")

    print("\n--- LOPO patches (patient_id >= 1) ---")
    lopo_mask = pids_np >= 1
    lopo_pids = pids_np[lopo_mask]
    lopo_lbls = lbls_np[lopo_mask]
    print(f"  Unique patients: {len(np.unique(lopo_pids))}")
    for name, idx in CLASS_MAPPING.items():
        print(f"  {name}: {int((lopo_lbls == idx).sum())}")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'all_images.npy'),      imgs_np)
    np.save(os.path.join(output_dir, 'all_labels.npy'),      lbls_np)
    np.save(os.path.join(output_dir, 'all_patient_ids.npy'), pids_np)

    print(f"\nSaved to '{output_dir}/':")
    print(f"  all_images.npy      {imgs_np.shape}  {imgs_np.dtype}")
    print(f"  all_labels.npy      {lbls_np.shape}  {lbls_np.dtype}")
    print(f"  all_patient_ids.npy {pids_np.shape}  {pids_np.dtype}")


if __name__ == "__main__":
    input_dir  = os.path.join(PROJECT_ROOT, "ILD_DB", "ILD_DB_talismanTestSuite")
    output_dir = os.path.join(PROJECT_ROOT, "patches", "ILD_DB_npy")
    prepare_lopo_dataset(input_dir, output_dir)
