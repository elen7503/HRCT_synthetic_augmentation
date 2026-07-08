"""
Diagnostic: why does the augmented dataset have zero synthetic
micronodules patches?

Run this and read the printed output top to bottom -- it stops being
useful past the first "FOUND ISSUE" line, since that's usually the
actual root cause and everything after it is a downstream symptom.
"""

import os
import numpy as np

REAL_DATA_DIR = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy/"
AUG_DATA_DIR = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/"
DIFFUSION_DIR = "/rds/general/user/eh1121/home/Final_Project/diffusion_model/"

CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]

print("=" * 70)
print("STEP 1: label encoding -- is micronodules really label index 4?")
print("=" * 70)
real_labels = np.load(os.path.join(REAL_DATA_DIR, "all_labels.npy"))
uniq, counts = np.unique(real_labels, return_counts=True)
print("Real dataset label distribution (label_value: count):")
for u, c in zip(uniq, counts):
    print(f"  {u}: {c}")
print(f"\nAssumed mapping: {dict(enumerate(CLASS_NAMES))}")
print("--> Check this against whatever produced labels.npy originally")
print("    (e.g. a class_to_idx dict in your preprocessing script).")
print("    If the counts here don't roughly match what you know about")
print("    each class's real patch count, the mapping is wrong -- FOUND ISSUE.\n")

print("=" * 70)
print("STEP 2: does a micronodules DDPM checkpoint even exist?")
print("=" * 70)
if os.path.isdir(DIFFUSION_DIR):
    for root, dirs, files in os.walk(DIFFUSION_DIR):
        for f in files:
            if "micronodul" in f.lower() or "class4" in f.lower() or "class_4" in f.lower():
                print(f"  found: {os.path.join(root, f)}")
    print("--> If nothing printed above, the micronodules model was never")
    print("    saved/trained to completion -- FOUND ISSUE: training never finished.\n")
else:
    print(f"  {DIFFUSION_DIR} not accessible from here -- check manually.\n")

print("=" * 70)
print("STEP 3: were any raw synthetic micronodules patches ever generated,")
print("        before merging into the augmented dataset?")
print("=" * 70)
print("Look for a directory like diffusion_model/generated_samples/ or")
print("similar, per-class. Manually run:")
print(f"  find {DIFFUSION_DIR} -iname '*micronodul*' -o -iname '*sample*micronodul*'")
print("--> If raw generated samples for micronodules exist on disk but")
print("    aren't in the augmented dataset, the bug is in the MERGE/")
print("    assembly script that builds ILD_DB_npy_augmented -- FOUND ISSUE:")
print("    generation worked, integration step dropped or mislabeled it.\n")

print("=" * 70)
print("STEP 4: augmented dataset -- full label x patient_id crosstab")
print("=" * 70)
aug_labels = np.load(os.path.join(AUG_DATA_DIR, "all_labels.npy"))
aug_pids = np.load(os.path.join(AUG_DATA_DIR, "all_patient_ids.npy"))
is_synth = aug_pids == -2
print("Synthetic patch count by label value:")
uniq, counts = np.unique(aug_labels[is_synth], return_counts=True)
for u, c in zip(uniq, counts):
    print(f"  label {u}: {c} synthetic patches")
print(f"\nTotal synthetic patches: {is_synth.sum()} / {len(aug_labels)} total rows")
print("--> If label 4 (or whatever micronodules maps to) is simply absent")
print("    from this list, it confirms zero synthetic patches exist for")
print("    that label in the FINAL merged file -- consistent with what")
print("    the quality script already showed you. This step is mainly")
print("    to catch a label-encoding mismatch (Step 1) vs. a true absence.\n")

print("=" * 70)
print("STEP 5: sanity check array lengths line up")
print("=" * 70)
aug_patches = np.load(os.path.join(AUG_DATA_DIR, "all_images.npy"), mmap_mode="r")
print(f"patches.npy:      {aug_patches.shape}")
print(f"labels.npy:       {aug_labels.shape}")
print(f"patient_ids.npy:  {aug_pids.shape}")
print("--> If these three lengths don't match, indexing is unreliable and")
print("    ANY per-class count (including the zero you saw) may be wrong --")
print("    FOUND ISSUE: misaligned arrays, fix before trusting any count.\n")