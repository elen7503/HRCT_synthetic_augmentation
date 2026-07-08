"""
DDPM Synthetic Patch Quality Assessment, per Tissue Class
===========================================================

Compares real vs. synthetic 32x32 HRCT patches for each of the 5 tissue
classes (healthy, emphysema, ground_glass, fibrosis, micronodules) using
several complementary metrics, to help decide whether the DDPM generator
is the bottleneck behind the flat classification results, or whether the
problem lies elsewhere (classifier, task design, evaluation setup).

METRICS COMPUTED PER CLASS
---------------------------
1. FID_classifier   Frechet Distance in the feature space of YOUR trained
                     CNN classifier (penultimate layer activations).
                     This is the recommended PRIMARY metric: it measures
                     quality using features actually learned on this data
                     distribution, at the resolution you care about,
                     instead of ImageNet-trained InceptionV3 features.

2. FID_inception     Standard FID using InceptionV3 pool features (patches
                     upsampled to 299x299, replicated to 3 channels).
                     Reported for comparability with the DDPM/GAN
                     literature (e.g. Lung-DDPM, 3D MedDiffusion) that
                     reports this metric -- treat it as a SECONDARY,
                     "for the lit-review table" number. At 32x32 input
                     resolution it mostly reflects upsampling artifacts
                     and ImageNet-domain feature mismatch, not necessarily
                     DDPM quality.

3. FID_pixel         Frechet Distance computed directly on flattened raw
                     pixel vectors (1024-dim). Crude, assumption-free
                     baseline that needs no pretrained network at all.

4. precision/recall  Kynkaanniemi et al. (2019) kNN-manifold precision and
                     recall, computed in classifier feature space.
                       precision ~ are synthetic samples realistic (fidelity)
                       recall    ~ do synthetic samples cover the real
                                   distribution (diversity)
                     A DDPM trained on very little data (e.g. emphysema,
                     ~1177 patches) can produce a handful of "safe",
                     average-looking samples that score a deceptively
                     decent FID while badly failing on recall (mode
                     collapse). FID alone cannot distinguish this from a
                     generator that is uniformly mediocre -- precision and
                     recall can.

5. wasserstein_px    1D Wasserstein distance between real and synthetic
                     pixel-intensity histograms. No feature extractor, no
                     assumptions -- a sanity-check floor metric.

All FID-type metrics are bootstrapped using EQUAL subsample sizes across
all 5 classes. This matters because FID is a biased estimator whose bias
shrinks as N grows -- since your classes have very different amounts of
real data (emphysema ~1177 vs. much larger classes), comparing raw
per-class FID computed on different N would partly just measure "how
much data did this class have available", confounding exactly the
correlation you want to test.

IMPORTANT CAVEAT -- READ BEFORE TRUSTING THE NUMBERS
------------------------------------------------------
If the "real" patches used as the reference set for a class are the same
patches that class's DDPM was trained on, an overfit/memorizing generator
(most likely for your smallest classes) can score artificially good FID
by nearly reproducing training patches. If you have a per-patient split
that lets you hold out patients NOT used to train a given class's DDMP,
set HOLDOUT_PATIENT_IDS below and use those as the real reference set.
If you don't have that available, report this as a known limitation of
the quality comparison in your dissertation -- it specifically inflates
apparent quality for the most data-starved classes, which is exactly the
comparison you're trying to make, so it's not a metric that will just
average out.

USAGE
-----
1. Edit the CONFIG block below to match your actual file layout.
2. python measure_ddpm_quality.py
3. Results land in ddpm_quality_results.csv and ddpm_quality_results.json
"""

import os
import json
import warnings
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import inception_v3, Inception_V3_Weights

# ============================================================
# CONFIG -- edit this block for your setup
# ============================================================

CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]
CLASS_LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}  # matches your 0-4 integer labels

# Real (non-augmented) dataset: expects patches.npy, labels.npy, patient_ids.npy
REAL_DATA_DIR = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy/"
REAL_PATCHES_FILE = os.path.join(REAL_DATA_DIR, "all_images.npy")
REAL_LABELS_FILE = os.path.join(REAL_DATA_DIR, "all_labels.npy")
REAL_PATIENT_IDS_FILE = os.path.join(REAL_DATA_DIR, "all_patient_ids.npy")

# Augmented dataset: same layout, synthetic patches flagged patient_id == -2
AUG_DATA_DIR = "/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/"
AUG_PATCHES_FILE = os.path.join(AUG_DATA_DIR, "all_images.npy")
AUG_LABELS_FILE = os.path.join(AUG_DATA_DIR, "all_labels.npy")
AUG_PATIENT_IDS_FILE = os.path.join(AUG_DATA_DIR, "all_patient_ids.npy")
SYNTHETIC_PATIENT_ID = -2

# If you can identify, per class, which real patients were held OUT of that
# class's DDPM training, list their patient_ids here to use as a cleaner
# (non-memorization-prone) FID reference set. Leave as None to just use all
# real patches of that class (simpler, but see caveat above).
HOLDOUT_PATIENT_IDS = None  # e.g. {"emphysema": [12, 45, 78], ...} or None

# Range your patches are stored in (BEFORE any DDPM-specific normalisation).
# Used to rescale to [0, 1] for feature extraction. Adjust if your patches
# are e.g. HU-windowed floats, or already in [-1, 1], or already [0, 1].
PATCH_VALUE_RANGE = (0.0, 1.0)  # (min, max) -- EDIT if this is wrong

# --- Classifier feature extraction (recommended primary metric) ---
USE_CLASSIFIER_FEATURES = True
# Path to your trained classifier checkpoint (state_dict). Set to None to skip.
CLASSIFIER_CHECKPOINT = "/rds/general/user/eh1121/home/Final_Project/classifier_checkpoint.pt"
# Import path to your model class -- EDIT to match your actual models.py location/name
# Expected to be a torch.nn.Module whose forward you can hook for penultimate features.
try:
    import sys
    sys.path.append("/rds/general/user/eh1121/home/Final_Project/ILD-Segmentation-And-Classification-DL/Supplementary_materials")
    from models import YourClassifierClass  # noqa: EDIT this import to your actual class name
except Exception as e:
    YourClassifierClass = None
    _IMPORT_ERROR = e

# Name of the layer whose output you want as the feature embedding.
# Run `print(model)` once to find the right name (e.g. "fc1", "classifier.0").
CLASSIFIER_FEATURE_LAYER = "fc1"  # EDIT to match your architecture

# --- Bootstrap settings ---
N_BOOTSTRAP = 30
RANDOM_SEED = 0
KNN_K = 5  # for precision/recall manifold estimation

OUT_CSV = "ddpm_quality_results.csv"
OUT_JSON = "ddpm_quality_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(RANDOM_SEED)

# ============================================================
# Data loading
# ============================================================

def load_class_patches(patches_file, labels_file, patient_ids_file, class_label,
                        patient_id_filter=None, patient_ids_whitelist=None):
    """
    Load patches for one class.
    patient_id_filter: if given, keep only rows where patient_id == this value
                        (used to select synthetic patches, id == -2).
    patient_ids_whitelist: if given, keep only rows whose patient_id is in this set
                        (used for the optional holdout real reference set).
    """
    patches = np.load(patches_file, mmap_mode="r")
    labels = np.load(labels_file)
    patient_ids = np.load(patient_ids_file)

    mask = labels == class_label
    if patient_id_filter is not None:
        mask &= (patient_ids == patient_id_filter)
    else:
        mask &= (patient_ids != SYNTHETIC_PATIENT_ID)  # exclude synthetic from "real"
        if patient_ids_whitelist is not None:
            mask &= np.isin(patient_ids, list(patient_ids_whitelist))

    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.empty((0, patches.shape[-2], patches.shape[-1]), dtype=np.float32)
    return np.asarray(patches[idx], dtype=np.float32)


def rescale_to_unit(x, value_range):
    lo, hi = value_range
    x = np.clip(x, lo, hi)
    return (x - lo) / max(hi - lo, 1e-8)


# ============================================================
# Feature extractors
# ============================================================

def extract_pixel_features(patches_01):
    """Flatten to raw pixel vectors. patches_01: (N, H, W) in [0,1]."""
    return patches_01.reshape(patches_01.shape[0], -1).astype(np.float64)


@torch.no_grad()
def extract_inception_features(patches_01, batch_size=64):
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=True)
    model.fc = nn.Identity()  # expose 2048-dim pool features
    model.eval().to(DEVICE)

    feats = []
    for i in range(0, len(patches_01), batch_size):
        batch = patches_01[i:i + batch_size]
        t = torch.from_numpy(batch).float().unsqueeze(1)  # (B,1,H,W)
        t = TF.resize(t, [299, 299], antialias=True)
        t = t.repeat(1, 3, 1, 1)  # grayscale -> 3-channel
        t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        t = t.to(DEVICE)
        out = model(t)
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float64)


def load_classifier_for_features():
    if YourClassifierClass is None:
        warnings.warn(
            f"Could not import your classifier class ({_IMPORT_ERROR}). "
            "Edit the import path/class name in CONFIG. Skipping classifier-space FID."
        )
        return None
    try:
        model = YourClassifierClass(num_classes=len(CLASS_NAMES))
        state = torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval().to(DEVICE)
        return model
    except Exception as e:
        warnings.warn(f"Could not load classifier checkpoint: {e}. Skipping classifier-space FID.")
        return None


_classifier_activation = {}

def _hook_fn(name):
    def hook(module, inp, out):
        _classifier_activation[name] = out.detach()
    return hook


@torch.no_grad()
def extract_classifier_features(patches_01, model, layer_name, batch_size=64):
    layer = dict(model.named_modules()).get(layer_name)
    if layer is None:
        raise ValueError(
            f"Layer '{layer_name}' not found. Run print(model) and fix "
            f"CLASSIFIER_FEATURE_LAYER in CONFIG. Available: {list(dict(model.named_modules()).keys())}"
        )
    handle = layer.register_forward_hook(_hook_fn(layer_name))
    feats = []
    try:
        for i in range(0, len(patches_01), batch_size):
            batch = patches_01[i:i + batch_size]
            t = torch.from_numpy(batch).float().unsqueeze(1).to(DEVICE)  # (B,1,H,W)
            _ = model(t)
            act = _classifier_activation[layer_name]
            feats.append(act.reshape(act.shape[0], -1).cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(feats, axis=0).astype(np.float64)


# ============================================================
# Frechet Distance
# ============================================================

def frechet_distance(feat_a, feat_b, eps=1e-6):
    mu1, mu2 = feat_a.mean(axis=0), feat_b.mean(axis=0)
    sigma1 = np.cov(feat_a, rowvar=False)
    sigma2 = np.cov(feat_b, rowvar=False)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def bootstrap_fid(feat_real, feat_synth, subsample_size, n_bootstrap=N_BOOTSTRAP):
    """FID with equal-size subsampling, repeated n_bootstrap times. Returns (mean, std)."""
    if len(feat_real) == 0 or len(feat_synth) == 0:
        return float("nan"), float("nan")
    vals = []
    for _ in range(n_bootstrap):
        idx_r = rng.choice(len(feat_real), size=subsample_size,
                            replace=len(feat_real) < subsample_size)
        idx_s = rng.choice(len(feat_synth), size=subsample_size,
                            replace=len(feat_synth) < subsample_size)
        vals.append(frechet_distance(feat_real[idx_r], feat_synth[idx_s]))
    return float(np.mean(vals)), float(np.std(vals))


# ============================================================
# Precision / Recall (Kynkaanniemi et al. 2019, simplified kNN version)
# ============================================================

def knn_radii(features, k=KNN_K):
    nn_model = NearestNeighbors(n_neighbors=k + 1).fit(features)
    dists, _ = nn_model.kneighbors(features)
    return dists[:, -1]  # distance to k-th neighbor, excluding self


def precision_recall(feat_real, feat_synth, k=KNN_K):
    if len(feat_real) < k + 1 or len(feat_synth) < k + 1:
        return float("nan"), float("nan")

    real_radii = knn_radii(feat_real, k)
    synth_radii = knn_radii(feat_synth, k)

    nn_real = NearestNeighbors(n_neighbors=1).fit(feat_real)
    dist_synth_to_real, idx_synth_to_real = nn_real.kneighbors(feat_synth)
    precision = np.mean(dist_synth_to_real[:, 0] <= real_radii[idx_synth_to_real[:, 0]])

    nn_synth = NearestNeighbors(n_neighbors=1).fit(feat_synth)
    dist_real_to_synth, idx_real_to_synth = nn_synth.kneighbors(feat_real)
    recall = np.mean(dist_real_to_synth[:, 0] <= synth_radii[idx_real_to_synth[:, 0]])

    return float(precision), float(recall)


# ============================================================
# Main
# ============================================================

@dataclass
class ClassResult:
    class_name: str
    n_real_available: int
    n_synthetic_available: int
    subsample_size: int
    FID_classifier_mean: float = float("nan")
    FID_classifier_std: float = float("nan")
    FID_inception_mean: float = float("nan")
    FID_inception_std: float = float("nan")
    FID_pixel_mean: float = float("nan")
    FID_pixel_std: float = float("nan")
    precision_classifier: float = float("nan")
    recall_classifier: float = float("nan")
    wasserstein_pixel_hist: float = float("nan")


def main():
    classifier_model = load_classifier_for_features() if USE_CLASSIFIER_FEATURES else None

    real_by_class, synth_by_class = {}, {}
    for cname in CLASS_NAMES:
        label = CLASS_LABEL_MAP[cname]
        whitelist = HOLDOUT_PATIENT_IDS.get(cname) if HOLDOUT_PATIENT_IDS else None
        real_by_class[cname] = load_class_patches(
            REAL_PATCHES_FILE, REAL_LABELS_FILE, REAL_PATIENT_IDS_FILE, label,
            patient_ids_whitelist=whitelist,
        )
        synth_by_class[cname] = load_class_patches(
            AUG_PATCHES_FILE, AUG_LABELS_FILE, AUG_PATIENT_IDS_FILE, label,
            patient_id_filter=SYNTHETIC_PATIENT_ID,
        )

    # Equal subsample size across ALL classes so comparisons aren't confounded
    # by different amounts of available data (see module docstring).
    min_n = min(
        min(len(v) for v in real_by_class.values() if len(v) > 0),
        min(len(v) for v in synth_by_class.values() if len(v) > 0),
    )
    subsample_size = max(min(min_n, 500), 10)  # cap at 500 for speed, floor at 10
    print(f"Using equal subsample size of {subsample_size} patches per class/condition "
          f"for all bootstrapped FID metrics.")

    results = []
    for cname in CLASS_NAMES:
        real_raw = real_by_class[cname]
        synth_raw = synth_by_class[cname]
        n_real, n_synth = len(real_raw), len(synth_raw)
        print(f"\n[{cname}] real={n_real} synthetic={n_synth}")

        if n_real == 0 or n_synth == 0:
            results.append(ClassResult(cname, n_real, n_synth, subsample_size))
            continue

        real_01 = rescale_to_unit(real_raw, PATCH_VALUE_RANGE)
        synth_01 = rescale_to_unit(synth_raw, PATCH_VALUE_RANGE)

        res = ClassResult(cname, n_real, n_synth, subsample_size)

        # --- pixel-space FID ---
        pf_real = extract_pixel_features(real_01)
        pf_synth = extract_pixel_features(synth_01)
        res.FID_pixel_mean, res.FID_pixel_std = bootstrap_fid(pf_real, pf_synth, subsample_size)

        # --- wasserstein on pixel intensity histograms ---
        res.wasserstein_pixel_hist = float(
            wasserstein_distance(real_01.ravel(), synth_01.ravel())
        )

        # --- inception FID ---
        try:
            if_real = extract_inception_features(real_01)
            if_synth = extract_inception_features(synth_01)
            res.FID_inception_mean, res.FID_inception_std = bootstrap_fid(
                if_real, if_synth, subsample_size
            )
        except Exception as e:
            warnings.warn(f"[{cname}] Inception FID failed: {e}")

        # --- classifier-space FID + precision/recall ---
        if classifier_model is not None:
            try:
                cf_real = extract_classifier_features(real_01, classifier_model, CLASSIFIER_FEATURE_LAYER)
                cf_synth = extract_classifier_features(synth_01, classifier_model, CLASSIFIER_FEATURE_LAYER)
                res.FID_classifier_mean, res.FID_classifier_std = bootstrap_fid(
                    cf_real, cf_synth, subsample_size
                )
                res.precision_classifier, res.recall_classifier = precision_recall(cf_real, cf_synth)
            except Exception as e:
                warnings.warn(f"[{cname}] classifier-space metrics failed: {e}")

        results.append(res)
        print(res)

    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\nSaved results to {OUT_CSV} and {OUT_JSON}")
    print(df.to_string(index=False))

    # Quick correlation check: does n_real_available track FID_classifier?
    valid = df.dropna(subset=["FID_classifier_mean"])
    if len(valid) >= 3:
        corr = np.corrcoef(valid["n_real_available"], valid["FID_classifier_mean"])[0, 1]
        print(f"\nCorrelation between n_real_available and FID_classifier_mean: {corr:.3f}")
        print("(Strongly negative => classes with less real data have worse/higher FID,")
        print(" supporting 'generator is the bottleneck for low-data classes'.")
        print(" Near zero => generator quality doesn't track with training set size;")
        print(" look elsewhere for the flat F1 result -- classifier capacity, patch size,")
        print(" evaluation leakage, or class-overlap at 32x32 resolution.)")


if __name__ == "__main__":
    main()