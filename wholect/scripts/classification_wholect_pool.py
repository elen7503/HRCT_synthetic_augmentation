"""
Patient-level classification via mean OR max pooling of per-slice CNN
features, on the FIXED stratified train/test split. Supports three
conditions: real-only, real+classic augmentation, real+synthetic.
"""

import os
import argparse

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import torch.optim as optim
from sklearn.metrics import f1_score

REAL_DATA_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_uint8")
AUGMENTED_DATA_DIR = os.path.join(PROJECT_ROOT, "wholect/ILD_DB_wholect_augmented")
CLASS_NAMES = ["healthy", "emphysema", "ground_glass", "fibrosis", "micronodules"]
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 15
LR = 1e-3


class WholeCTPoolClassifier(nn.Module):
    def __init__(self, num_classes=5, pooling="mean"):
        super().__init__()
        self.pooling = pooling

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )
        self.features = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64),
            block(64, 128), block(128, 128), block(128, 128),
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, bag):
        feats = self.features(bag)
        feats = feats.view(feats.size(0), -1)
        if self.pooling == "mean":
            pooled = feats.mean(dim=0, keepdim=True)
        else:
            pooled = feats.max(dim=0, keepdim=True).values
        x = self.drop(TF.relu(self.fc1(pooled)))
        return self.fc2(x)


def augment_slice(img):
    k = np.random.randint(0, 4)
    img = torch.rot90(img, k, dims=[1, 2])
    if np.random.rand() < 0.5:
        img = torch.flip(img, dims=[2])
    if np.random.rand() < 0.5:
        img = torch.flip(img, dims=[1])
    return img


def build_bags(pids_to_use, all_images, all_labels, all_pids):
    bags = {}
    for pid in pids_to_use:
        mask = all_pids == pid
        if not mask.any():
            continue
        imgs = all_images[mask]
        lbls = all_labels[mask]
        vals, counts = np.unique(lbls, return_counts=True)
        label = int(vals[np.argmax(counts)])
        bags[int(pid)] = (imgs, label)
    return bags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["real", "real_aug", "real_synth"], required=True)
    parser.add_argument("--pooling", choices=["mean", "max"], required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_pids = np.load(os.path.join(REAL_DATA_DIR, f"fixed_split_train_pids_seed{args.split_seed}.npy"))
    test_pids = np.load(os.path.join(REAL_DATA_DIR, f"fixed_split_test_pids_seed{args.split_seed}.npy"))

    real_images = np.load(os.path.join(REAL_DATA_DIR, "all_images.npy"))
    real_labels = np.load(os.path.join(REAL_DATA_DIR, "all_labels.npy"))
    real_pids = np.load(os.path.join(REAL_DATA_DIR, "all_patient_ids.npy"))

    train_bags = build_bags(train_pids, real_images, real_labels, real_pids)
    test_bags = build_bags(test_pids, real_images, real_labels, real_pids)

    if args.condition == "real_synth":
        aug_images = np.load(os.path.join(AUGMENTED_DATA_DIR, "all_images.npy"))
        aug_pids = np.load(os.path.join(AUGMENTED_DATA_DIR, "all_patient_ids.npy"))
        n_real = len(real_images)
        synth_images = aug_images[n_real:]
        synth_pids = aug_pids[n_real:]

        added = 0
        for pid in train_pids:
            mask = synth_pids == int(pid)
            if mask.any():
                extra_imgs = synth_images[mask]
                imgs, label = train_bags[int(pid)]
                train_bags[int(pid)] = (np.concatenate([imgs, extra_imgs], axis=0), label)
                added += mask.sum()
        print(f"Added {added} synthetic slices to {len(train_pids)} training patients' bags "
              f"(sentinel-sourced synthetic slices excluded)")

    net = WholeCTPoolClassifier(NUM_CLASSES, pooling=args.pooling).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    train_pid_list = list(train_bags.keys())
    print(f"Training: {len(train_pid_list)} patient bags | condition={args.condition} pooling={args.pooling}")

    for epoch in range(NUM_EPOCHS):
        net.train()
        np.random.shuffle(train_pid_list)
        epoch_loss = 0.0
        for pid in train_pid_list:
            imgs, label = train_bags[pid]
            x = torch.from_numpy(imgs.astype(np.float32) / 255.0).unsqueeze(1).to(DEVICE)
            if args.condition == "real_aug":
                x = torch.stack([augment_slice(x[i]) for i in range(x.shape[0])])
            y = torch.tensor([label], dtype=torch.long).to(DEVICE)

            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS} | avg bag loss: {epoch_loss/len(train_pid_list):.4f}")

    net.eval()
    preds, trues = [], []
    with torch.no_grad():
        for pid, (imgs, label) in test_bags.items():
            x = torch.from_numpy(imgs.astype(np.float32) / 255.0).unsqueeze(1).to(DEVICE)
            out = net(x)
            pred = int(out.argmax(dim=1).item())
            preds.append(pred)
            trues.append(label)
            print(f"  Test patient {pid}: pred={CLASS_NAMES[pred]} true={CLASS_NAMES[label]} "
                  f"{'CORRECT' if pred == label else 'WRONG'}")

    f1_per_class = f1_score(trues, preds, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    macro_f1 = float(np.mean(f1_per_class))
    accuracy = float(np.mean(np.array(preds) == np.array(trues)))

    print("\n" + "=" * 60)
    print(f"FIXED-SPLIT PATIENT-LEVEL Summary -- condition={args.condition} pooling={args.pooling} seed={args.seed}")
    print("=" * 60)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15} F1 = {f1_per_class[i]:.4f}")
    print(f"  {'Macro (5 classes)':<15} F1 = {macro_f1:.4f}")
    print(f"  {'Patient accuracy':<15} = {accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
