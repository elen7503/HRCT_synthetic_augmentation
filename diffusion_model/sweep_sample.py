"""
Samples from EVERY saved checkpoint for one class's DDPM, saving each checkpoint's output into its
own subfolder.
"""

import os
import glob
import argparse
import torch

from ddpm import UNet, DDPM
from dataset import CLASS_NAMES, HU_MIN, HU_MAX
import numpy as np


def denormalise(x):
    x = (x + 1.0) / 2.0
    x = x * (HU_MAX - HU_MIN) + HU_MIN
    return x


@torch.no_grad()
def sample_one_checkpoint(ckpt_path, class_idx, n_samples, batch_size, timesteps, device):
    unet = UNet(channels=64, time_dim=256).to(device)
    ddpm = DDPM(unet, timesteps=timesteps).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ddpm.load_state_dict(ckpt['model_state_dict'])
    ddpm.eval()

    all_samples = []
    generated = 0
    while generated < n_samples:
        n = min(batch_size, n_samples - generated)
        samples = ddpm.sample(n, device, img_size=32)
        samples = samples.squeeze(1).cpu().numpy()
        samples = denormalise(samples).astype(np.int16)
        all_samples.append(samples)
        generated += n

    return np.concatenate(all_samples, axis=0), ckpt.get('epoch', -1), ckpt.get('loss', float('nan'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_idx', type=int, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Folder containing best.pt and epoch_*.pt files, e.g. outputs/fibrosis')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Kept modest since this runs once per checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_name = CLASS_NAMES[args.class_idx]

    ckpt_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'epoch_*.pt')))
    best_path = os.path.join(args.checkpoint_dir, 'best.pt')
    if os.path.exists(best_path):
        ckpt_paths.append(best_path)

    if not ckpt_paths:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    print(f"Found {len(ckpt_paths)} checkpoints for {class_name}: {[os.path.basename(p) for p in ckpt_paths]}")
    os.makedirs(args.output_dir, exist_ok=True)

    summary = []
    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        print(f"\nSampling from {ckpt_name}...")
        samples, epoch, loss = sample_one_checkpoint(
            ckpt_path, args.class_idx, args.n_samples, args.batch_size, args.timesteps, device
        )
        out_dir = os.path.join(args.output_dir, ckpt_name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'synthetic_images.npy'), samples)
        print(f"  epoch={epoch}  loss={loss:.4f}  saved {len(samples)} samples to {out_dir}")
        summary.append({'checkpoint': ckpt_name, 'epoch': epoch, 'train_loss': loss, 'dir': out_dir})

    import json
    with open(os.path.join(args.output_dir, 'sweep_manifest.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nManifest saved to {os.path.join(args.output_dir, 'sweep_manifest.json')}")


if __name__ == "__main__":
    main()