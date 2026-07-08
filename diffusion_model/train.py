"""
train.py
--------
Trains a separate DDPM for each tissue class (or a single specified class).

Usage:
  python train.py --class_idx 3 --epochs 500 --batch_size 64
"""

import os
import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ddpm import UNet, DDPM
from dataset import get_dataloader, CLASS_NAMES


def train(class_idx, npy_dir, output_dir, epochs, batch_size, lr, timesteps, save_every):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Training class: {CLASS_NAMES[class_idx]} (idx={class_idx})")

    # ── Data ─────────────────────────────────────────────────────────────────
    loader = get_dataloader(npy_dir, class_idx, batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    unet = UNet(channels=64, time_dim=256).to(device)
    ddpm = DDPM(unet, timesteps=timesteps).to(device)

    n_params = sum(p.numel() for p in unet.parameters())
    print(f"U-Net parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    opt = Adam(ddpm.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    # ── Output dir ────────────────────────────────────────────────────────────
    class_name = CLASS_NAMES[class_idx]
    ckpt_dir = os.path.join(output_dir, class_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        ddpm.train()
        epoch_loss = 0.0
        n_batches = 0

        for x in loader:
            x = x.to(device)
            loss = ddpm.loss(x)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{epochs} | loss: {avg_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddpm.state_dict(),
                'loss': best_loss,
                'class_idx': class_idx,
                'class_name': class_name,
            }, os.path.join(ckpt_dir, 'best.pt'))

        # Save periodic checkpoint
        if epoch % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddpm.state_dict(),
                'loss': avg_loss,
                'class_idx': class_idx,
                'class_name': class_name,
            }, os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pt'))

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_idx',  type=int,   default=3,
                        help='Class index to train (0=healthy,1=emphysema,2=ground_glass,3=fibrosis,4=micronodules)')
    parser.add_argument('--npy_dir',    type=str,   default='/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy')
    parser.add_argument('--output_dir', type=str,   default='/rds/general/user/eh1121/home/Final_Project/diffusion_model/outputs')
    parser.add_argument('--epochs',     type=int,   default=500)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--timesteps',  type=int,   default=1000)
    parser.add_argument('--save_every', type=int,   default=100)
    args = parser.parse_args()

    train(
        class_idx=args.class_idx,
        npy_dir=args.npy_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        save_every=args.save_every,
    )
