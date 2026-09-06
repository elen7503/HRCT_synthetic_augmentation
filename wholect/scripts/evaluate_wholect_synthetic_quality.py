"""
Same FID (classifier feature space) + precision/recall approach used for
the patch-level DDPMs, adapted for whole-CT slices. Compares real
ILD_DB_wholect_npy patches against Lung-DDPM-generated samples, per class.
"""

import os
import glob

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
import torch

from classification_wholect_lopo import WholeCTClassifier, NUM_CLASSES, CLASS_NAMES, DEVICE

REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_npy")  # only used for feature extractor path now
FEATURE_EXTRACTOR_PATH = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_npy/feature_extractor_lungddpm_domain.pth")

ILD_DATASET_DIR = os.path.join(PROJECT_ROOT, "wholect/lung_ddpm_model/ILD_dataset")
VALIDATION_MANIFEST_PATH = os.path.join(PROJECT_ROOT, "wholect/lung_ddpm_model/Lung-DDPM/checkpoints/ILD-DDPM-2D-V3/validation_manifest.json")

GENERATED_DIR = os.path.join(PROJECT_ROOT, "wholect/lung_ddpm_model/Lung-DDPM/generated/ILD-DDPM-2D-V3-balanced-test")

KNN_K = 5
N_BOOTSTRAP = 30
RANDOM_SEED = 0
rng = np.random.default_rng(RANDOM_SEED)
OUT_CSV = "wholect_synthetic_quality_results.csv"


def apply_lung_window(hu_image, center=-600, width=1600):
    """Same transform used in train_wholect_feature_extractor.py -- must
    stay identical so the classifier sees the same intensity scale it was
    trained on."""
    lo = center - width / 2
    hi = center + width / 2
    clipped = np.clip(hu_image, lo, hi).astype(np.float32)
    return (clipped - lo) / (hi - lo)


def load_classifier():
    model = WholeCTClassifier(NUM_CLASSES)
    ckpt = torch.load(FEATURE_EXTRACTOR_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(DEVICE)
    return model


def normalize_pair(real, synth):
    real_norm = real.astype(np.float32) / 255.0
    synth_norm = synth.astype(np.float32) / 255.0
    return real_norm, synth_norm


_activation = {}
def _hook(name):
    def fn(module, inp, out):
        _activation[name] = out.detach()
    return fn


@torch.no_grad()
def extract_features(images, model, batch_size=8):
    """Uses the penultimate layer (before fc2) as the feature space."""
    layer = model.fc1
    handle = layer.register_forward_hook(_hook("fc1"))
    feats = []
    try:
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            t = torch.from_numpy(batch).float().unsqueeze(1).to(DEVICE)
            _ = model(t)
            feats.append(_activation["fc1"].cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(feats, axis=0).astype(np.float64)


def frechet_distance(a, b, eps=1e-6):
    mu1, mu2 = a.mean(axis=0), b.mean(axis=0)
    s1, s2 = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((s1 + offset).dot(s2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def bootstrap_fid(feat_a, feat_b, n_sub, n_bootstrap=N_BOOTSTRAP):
    vals = []
    for _ in range(n_bootstrap):
        ia = rng.choice(len(feat_a), size=n_sub, replace=len(feat_a) < n_sub)
        ib = rng.choice(len(feat_b), size=n_sub, replace=len(feat_b) < n_sub)
        vals.append(frechet_distance(feat_a[ia], feat_b[ib]))
    return float(np.mean(vals)), float(np.std(vals))


def precision_recall(feat_real, feat_synth, k=KNN_K):
    if len(feat_real) < k + 1 or len(feat_synth) < k + 1:
        return float("nan"), float("nan")
    real_radii = NearestNeighbors(n_neighbors=k + 1).fit(feat_real).kneighbors(feat_real)[0][:, -1]
    synth_radii = NearestNeighbors(n_neighbors=k + 1).fit(feat_synth).kneighbors(feat_synth)[0][:, -1]
    nn_real = NearestNeighbors(n_neighbors=1).fit(feat_real)
    d_s2r, idx_s2r = nn_real.kneighbors(feat_synth)
    precision = np.mean(d_s2r[:, 0] <= real_radii[idx_s2r[:, 0]])
    nn_synth = NearestNeighbors(n_neighbors=1).fit(feat_synth)
    d_r2s, idx_r2s = nn_synth.kneighbors(feat_real)
    recall = np.mean(d_r2s[:, 0] <= synth_radii[idx_r2s[:, 0]])
    return float(precision), float(recall)


import json

LUNGDDPM_LABEL_TO_YOUR_CLASS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}


def load_real_class(class_idx):
    lungddpm_label = None
    for l, c in LUNGDDPM_LABEL_TO_YOUR_CLASS.items():
        if c == class_idx:
            lungddpm_label = l
            break
    if lungddpm_label is None:
        return None

    with open(VALIDATION_MANIFEST_PATH) as f:
        val_manifest = json.load(f)

    images = []
    for record in val_manifest["records"]:
        if lungddpm_label in record["labels"]:
            img_path = os.path.join(ILD_DATASET_DIR, record["files"]["image"]["path"])
            if os.path.exists(img_path):
                images.append(np.load(img_path))

    if not images:
        return None
    return np.stack(images)


def load_generated_class(class_idx, class_name):
    report_path = os.path.join(GENERATED_DIR, "generation_report.json")
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        report = json.load(f)

    lungddpm_label = None
    for l, c in LUNGDDPM_LABEL_TO_YOUR_CLASS.items():
        if c == class_idx:
            lungddpm_label = l
            break
    if lungddpm_label is None:
        return None  # class not in the DDPM's scheme (shouldn't happen for your 5)

    matching_cases = [c["case"] for c in report["cases"] if c["target_class"] == lungddpm_label]
    if not matching_cases:
        return None

    images = []
    for case in matching_cases:
        case_dir = os.path.join(GENERATED_DIR, case, "images")
        npy_files = sorted(glob.glob(os.path.join(case_dir, "*.npy")))
        for f in npy_files:
            arr = np.load(f)
            if arr.shape != (512, 512):
                import cv2
                arr = cv2.resize(arr.astype(np.float32), (512, 512),
                                 interpolation=cv2.INTER_LINEAR)
            images.append(arr)

    if not images:
        return None
    return np.stack(images)


def main():
    model = load_classifier()

    results = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_name_lower = class_name.lower().replace(" ", "_")
        real_patches = load_real_class(class_idx)
        synth_patches = load_generated_class(class_idx, class_name_lower)

        if real_patches is None:
            print(f"[SKIP] {class_name}: no real validation slices found for this class")
            continue
        if synth_patches is None:
            print(f"[SKIP] {class_name}: no generated samples found")
            continue

        real_norm, synth_norm = normalize_pair(real_patches, synth_patches)
        feat_real = extract_features(real_norm, model)
        feat_synth = extract_features(synth_norm, model)

        n_sub = min(len(feat_real), len(feat_synth), 100)
        fid_mean, fid_std = bootstrap_fid(feat_real, feat_synth, n_sub)
        precision, recall = precision_recall(feat_real, feat_synth)

        print(f"{class_name:15s}  n_real={len(real_patches):4d}  n_synth={len(synth_patches):4d}  "
              f"FID={fid_mean:8.2f}±{fid_std:.2f}  precision={precision:.3f}  recall={recall:.3f}")

        results.append({
            "class_name": class_name, "n_real": len(real_patches), "n_synth": len(synth_patches),
            "FID_mean": fid_mean, "FID_std": fid_std, "precision": precision, "recall": recall,
        })

    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved to {OUT_CSV}")
    else:
        print("\nNo results -- check GENERATED_DIR and load_generated_class() match the actual output.")


if __name__ == "__main__":
    main()