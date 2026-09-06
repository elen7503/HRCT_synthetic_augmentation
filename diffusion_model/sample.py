"""
Generates synthetic patches from a trained DDPM checkpoint.
"""

import os
import argparse

PROJECT_ROOT = os.environ["PROJECT_ROOT"]
import numpy as np
import torch

from ddpm import UNet, DDPM
from dataset import CLASS_NAMES, HU_MIN, HU_MAX


def denormalise(x):
    """Convert [-1, 1] back to HU values."""
    x = (x + 1.0) / 2.0 
    x = x * (HU_MAX - HU_MIN) + HU_MIN 
    return x


def sample(class_idx, weightfile, output_dir, n_samples, batch_size, timesteps):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Sampling class: {CLASS_NAMES[class_idx]} (idx={class_idx})")
    print(f"Checkpoint: {weightfile}")

    unet = UNet(channels=64, time_dim=256).to(device)
    ddpm = DDPM(unet, timesteps=timesteps).to(device)

    ckpt = torch.load(weightfile, map_location=device)
    ddpm.load_state_dict(ckpt['model_state_dict'])
    ddpm.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")

    all_samples = []
    generated = 0

    while generated < n_samples:
        n = min(batch_size, n_samples - generated)
        samples = ddpm.sample(n, device, img_size=32)
        samples = samples.squeeze(1).cpu().numpy()
        samples = denormalise(samples).astype(np.int16)
        all_samples.append(samples)
        generated += n
        print(f"  Generated {generated}/{n_samples}")

    all_samples = np.concatenate(all_samples, axis=0)
    all_labels  = np.full(n_samples, class_idx, dtype=np.int64)

    class_name = CLASS_NAMES[class_idx]
    os.makedirs(output_dir, exist_ok=True)
    out_images = os.path.join(output_dir, f"synthetic_{class_name}_images.npy")
    out_labels = os.path.join(output_dir, f"synthetic_{class_name}_labels.npy")

    np.save(out_images, all_samples)
    np.save(out_labels, all_labels)

    print(f"\nSaved {n_samples} synthetic patches:")
    print(f"  {out_images}  {all_samples.shape}  {all_samples.dtype}")
    print(f"  {out_labels}  {all_labels.shape}   {all_labels.dtype}")
    print(f"  HU range: [{all_samples.min()}, {all_samples.max()}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_idx',  type=int,   required=True,
                        help='Class index (0=healthy,1=emphysema,2=ground_glass,3=fibrosis,4=micronodules)')
    parser.add_argument('--weightfile', type=str,   required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--output_dir', type=str,   default=os.path.join(PROJECT_ROOT, "diffusion_model/outputs/synthetic"))
    parser.add_argument('--n_samples',  type=int,   default=500)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--timesteps',  type=int,   default=1000)
    args = parser.parse_args()

    sample(
        class_idx=args.class_idx,
        weightfile=args.weightfile,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        timesteps=args.timesteps,
    )
