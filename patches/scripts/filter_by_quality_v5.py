"""
Filters synthetic patches per class using two criteria
"""

import os
import sys

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, cdist

sys.path.append(os.path.join(PROJECT_ROOT, "classifier_lib/Lung_Classification"))
from models import Classifier

CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]

POOL_DIR = os.path.join(PROJECT_ROOT, "diffusion_model/outputs/synthetic_v5_pool")
OUT_DIR = os.path.join(PROJECT_ROOT, "diffusion_model/outputs/synthetic_v5")

CLASSIFIER_CHECKPOINT = os.path.join(PROJECT_ROOT, "patches/experiments/lopo_baseline_v4_seed0_20260801_075708/checkpoints/fold_001_pid_1_final.pth")
CLASSIFIER_FEATURE_LAYER = "fc1"

FIXED_DOSE = 5000
MIN_DIST_PERCENTILE = 10
RANDOM_SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(RANDOM_SEED)

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


def normalize_for_classifier(patches):
    img_min = float(patches.min())
    img_max = float(patches.max())
    denom = img_max - img_min + 1e-8
    return (patches.astype(np.float32) - img_min) / denom


@torch.no_grad()
def score_and_extract(patches, model, class_idx, layer_name, batch_size=128):
    layer = dict(model.named_modules()).get(layer_name)
    handle = layer.register_forward_hook(_hook(layer_name))
    confidences, features = [], []
    try:
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            t = torch.from_numpy(batch).float().unsqueeze(1).to(DEVICE)
            logits = model(t)
            probs = F.softmax(logits, dim=1)
            confidences.append(probs[:, class_idx].cpu().numpy())
            act = _activation[layer_name]
            features.append(act.reshape(act.shape[0], -1).cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(confidences), np.concatenate(features).astype(np.float64)


def greedy_diverse_selection(confidences, features, n_keep, min_dist):
    ranked = np.argsort(confidences)[::-1]
    kept_idx = []
    kept_feats = []

    for idx in ranked:
        if len(kept_idx) >= n_keep:
            break
        candidate = features[idx:idx+1]
        if not kept_feats:
            kept_idx.append(idx)
            kept_feats.append(features[idx])
            continue
        dists = cdist(candidate, np.array(kept_feats))[0]
        if dists.min() > min_dist:
            kept_idx.append(idx)
            kept_feats.append(features[idx])

    n_from_diverse_pass = len(kept_idx)
    relaxed = False

    if len(kept_idx) < n_keep:
        relaxed = True
        already_kept_set = set(kept_idx)
        for idx in ranked:
            if len(kept_idx) >= n_keep:
                break
            if idx not in already_kept_set:
                kept_idx.append(idx)

    return np.array(kept_idx[:n_keep]), n_from_diverse_pass, relaxed


def diversity_check(features, kept_idx, n_random_samples=500):
    kept_feats = features[kept_idx]
    n_sub = min(len(kept_feats), n_random_samples)

    sub_kept = kept_feats[rng.choice(len(kept_feats), size=n_sub, replace=False)]
    mean_dist_kept = pdist(sub_kept).mean()

    random_idx = rng.choice(len(features), size=n_sub, replace=False)
    sub_random = features[random_idx]
    mean_dist_random = pdist(sub_random).mean()

    return mean_dist_kept, mean_dist_random


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_classifier()

    for class_idx, class_name in enumerate(CLASS_NAMES):
        img_path = os.path.join(POOL_DIR, f"synthetic_{class_name}_images.npy")
        if not os.path.exists(img_path):
            print(f"[SKIP] {class_name}: no pool file found")
            continue

        pool_images = np.load(img_path)
        real_images_this_class = np.load(
            os.path.join(PROJECT_ROOT, "ILD_DB_npy/all_images.npy")
        )[np.load(
            os.path.join(PROJECT_ROOT, "ILD_DB_npy/all_labels.npy")
        ) == class_idx]
        img_min = float(real_images_this_class.min())
        img_max = float(real_images_this_class.max())
        denom = img_max - img_min + 1e-8
        pool_norm = ((pool_images.astype(np.float32) - img_min) / denom)
        confidences, features = score_and_extract(pool_norm, model, class_idx, CLASSIFIER_FEATURE_LAYER)

        sample_pairs = pdist(features[rng.choice(len(features), size=min(500, len(features)), replace=False)])
        min_dist = np.percentile(sample_pairs, MIN_DIST_PERCENTILE)

        n_keep = min(FIXED_DOSE, len(pool_images))
        kept_idx, n_from_diverse_pass, relaxed = greedy_diverse_selection(
            confidences, features, n_keep, min_dist
        )

        mean_dist_kept, mean_dist_random = diversity_check(features, kept_idx)

        filtered_images = pool_images[kept_idx]
        filtered_labels = np.full(len(kept_idx), class_idx, dtype=np.int64)

        flag = " [DIVERSITY WARNING: kept set less diverse than random pool sample]" \
               if mean_dist_kept < mean_dist_random else ""
        relax_note = f" (relaxed to fill {n_keep - n_from_diverse_pass} slots by confidence alone)" if relaxed else ""

        print(f"{class_name:15s}  pool={len(pool_images)}  kept={len(kept_idx)}  "
              f"mean_conf(kept)={confidences[kept_idx].mean():.3f}  "
              f"diverse_pass={n_from_diverse_pass}{relax_note}")
        print(f"{'':15s}  mean_pairwise_dist: kept={mean_dist_kept:.2f}  "
              f"random_sample={mean_dist_random:.2f}{flag}")

        np.save(os.path.join(OUT_DIR, f"synthetic_{class_name}_images.npy"), filtered_images)
        np.save(os.path.join(OUT_DIR, f"synthetic_{class_name}_labels.npy"), filtered_labels)

    print(f"\nFiltered synthetic patches saved to {OUT_DIR}/")
    print("If any class shows a DIVERSITY WARNING, the filter is still net-reducing")
    print("diversity for that class relative to random sampling -- flag this explicitly")
    print("rather than treating the filtered set as an unambiguous improvement.")


if __name__ == "__main__":
    main()