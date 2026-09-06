"""
sweep_evaluate_all_classes.py
--------------------------------
Same as sweep_evaluate.py, but loops over all 5 classes automatically
and writes one combined CSV, instead of editing/rerunning per class.

Usage:
  1. Edit CLASSIFIER_CHECKPOINT below.
  2. python sweep_evaluate_all_classes.py
"""

import os
import sys
import json

PROJECT_ROOT = os.environ["PROJECT_ROOT"]

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.neighbors import NearestNeighbors

import torch

sys.path.append(os.path.join(PROJECT_ROOT, "classifier_lib/Lung_Classification"))
from models import Classifier

# ============================================================
CLASS_NAMES = {0: "healthy", 1: "emphysema", 2: "ground_glass", 3: "fibrosis", 4: "micronodules"}

REAL_IMGS_PATH = os.path.join(PROJECT_ROOT, "ILD_DB_npy/all_images.npy")
REAL_LBLS_PATH = os.path.join(PROJECT_ROOT, "ILD_DB_npy/all_labels.npy")

SWEEP_BASE_DIR = os.path.join(PROJECT_ROOT, "diffusion_model/outputs_attention/checkpoint_sweep")

# EDIT: point at any real fold checkpoint
CLASSIFIER_CHECKPOINT = os.path.join(PROJECT_ROOT, "patches/checkpoints/feature_extractor_reference.pth")
CLASSIFIER_FEATURE_LAYER = "fc1"

KNN_K = 5
N_BOOTSTRAP = 30
RANDOM_SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(RANDOM_SEED)
OUT_CSV = "checkpoint_sweep_attention_results.csv"
# ============================================================

_activation = {}

def _hook(name):
    def fn(module, inp, out):
        _activation[name] = out.detach()
    return fn


def load_classifier():
    model = Classifier(num_classes=5)
    state = torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE, weights_only=False)
    sd = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(sd)
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def extract_features(patches, model, layer_name, batch_size=64):
    layer = dict(model.named_modules()).get(layer_name)
    handle = layer.register_forward_hook(_hook(layer_name))
    feats = []
    try:
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            t = torch.from_numpy(batch).float().unsqueeze(1).to(DEVICE)
            _ = model(t)
            act = _activation[layer_name]
            feats.append(act.reshape(act.shape[0], -1).cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(feats, axis=0).astype(np.float64)


def classifier_normalize(real_patches, synth_patches):
    img_min = min(float(real_patches.min()), float(synth_patches.min()))
    img_max = max(float(real_patches.max()), float(synth_patches.max()))
    denom = img_max - img_min + 1e-8
    real_norm = (real_patches.astype(np.float32) - img_min) / denom
    synth_norm = (synth_patches.astype(np.float32) - img_min) / denom
    return real_norm, synth_norm


def frechet_distance(feat_a, feat_b, eps=1e-6):
    mu1, mu2 = feat_a.mean(axis=0), feat_b.mean(axis=0)
    sigma1 = np.atleast_2d(np.cov(feat_a, rowvar=False))
    sigma2 = np.atleast_2d(np.cov(feat_b, rowvar=False))
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def bootstrap_fid(feat_real, feat_synth, n_sub, n_bootstrap=N_BOOTSTRAP):
    vals = []
    for _ in range(n_bootstrap):
        idx_r = rng.choice(len(feat_real), size=n_sub, replace=len(feat_real) < n_sub)
        idx_s = rng.choice(len(feat_synth), size=n_sub, replace=len(feat_synth) < n_sub)
        vals.append(frechet_distance(feat_real[idx_r], feat_synth[idx_s]))
    return float(np.mean(vals)), float(np.std(vals))


def knn_radii(features, k=KNN_K):
    nn_model = NearestNeighbors(n_neighbors=k + 1).fit(features)
    dists, _ = nn_model.kneighbors(features)
    return dists[:, -1]


def precision_recall(feat_real, feat_synth, k=KNN_K):
    if len(feat_real) < k + 1 or len(feat_synth) < k + 1:
        return float("nan"), float("nan")
    real_radii = knn_radii(feat_real, k)
    synth_radii = knn_radii(feat_synth, k)
    nn_real = NearestNeighbors(n_neighbors=1).fit(feat_real)
    d_s2r, idx_s2r = nn_real.kneighbors(feat_synth)
    precision = np.mean(d_s2r[:, 0] <= real_radii[idx_s2r[:, 0]])
    nn_synth = NearestNeighbors(n_neighbors=1).fit(feat_synth)
    d_r2s, idx_r2s = nn_synth.kneighbors(feat_real)
    recall = np.mean(d_r2s[:, 0] <= synth_radii[idx_r2s[:, 0]])
    return float(precision), float(recall)


def evaluate_class(class_idx, class_name, model, real_images, real_labels):
    sweep_dir = os.path.join(SWEEP_BASE_DIR, class_name)
    manifest_path = os.path.join(sweep_dir, 'sweep_manifest.json')
    if not os.path.exists(manifest_path):
        print(f"[SKIP] {class_name}: no manifest found at {manifest_path}")
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    real_patches = real_images[real_labels == class_idx]
    results = []

    for entry in manifest:
        # entry['dir'] may be relative (bug in sweep_sample.py) -- try both
        synth_path = os.path.join(entry['dir'], 'synthetic_images.npy')
        if not os.path.exists(synth_path):
            synth_path = os.path.join(sweep_dir, entry['checkpoint'], 'synthetic_images.npy')
        if not os.path.exists(synth_path):
            print(f"  [SKIP] {class_name}/{entry['checkpoint']}: file not found")
            continue

        synth_patches = np.load(synth_path).astype(np.float32)
        real_norm, synth_norm = classifier_normalize(real_patches, synth_patches)
        feat_real = extract_features(real_norm, model, CLASSIFIER_FEATURE_LAYER)
        feat_synth = extract_features(synth_norm, model, CLASSIFIER_FEATURE_LAYER)

        n_sub = min(len(feat_real), len(feat_synth), 100)
        fid_mean, fid_std = bootstrap_fid(feat_real, feat_synth, n_sub)
        precision, recall = precision_recall(feat_real, feat_synth)

        print(f"{class_name:15s} {entry['checkpoint']:15s}  epoch={entry['epoch']:5}  "
              f"train_loss={entry['train_loss']:.4f}  FID={fid_mean:8.2f}±{fid_std:.2f}  "
              f"precision={precision:.3f}  recall={recall:.3f}")

        results.append({
            "class_name": class_name,
            "checkpoint": entry['checkpoint'],
            "epoch": entry['epoch'],
            "train_loss": entry['train_loss'],
            "FID_mean": fid_mean,
            "FID_std": fid_std,
            "precision": precision,
            "recall": recall,
        })

    return results


def main():
    model = load_classifier()
    real_images = np.load(REAL_IMGS_PATH).astype(np.float32)
    real_labels = np.load(REAL_LBLS_PATH)

    all_results = []
    for class_idx, class_name in CLASS_NAMES.items():
        all_results.extend(evaluate_class(class_idx, class_name, model, real_images, real_labels))

    df = pd.DataFrame(all_results).sort_values(["class_name", "recall"], ascending=[True, False])
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")
    print(df.to_string(index=False))

    print("\nBest checkpoint per class (highest recall):")
    best_per_class = df.loc[df.groupby("class_name")["recall"].idxmax()]
    print(best_per_class[["class_name", "checkpoint", "epoch", "train_loss", "FID_mean", "precision", "recall"]].to_string(index=False))


if __name__ == "__main__":
    main()