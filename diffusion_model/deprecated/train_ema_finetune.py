"""
train_ema_finetune.py
------------------------
Resumes DDPM training from an existing checkpoint (e.g. outputs/fibrosis/best.pt)
for a modest number of additional epochs, while maintaining an EMA (exponential
moving average) copy of the model weights. Samples generated from the EMA copy
are typically less noisy than samples from raw end-of-training weights -- this
is standard practice in DDPM training (including Ho et al. 2020) and was
missing from the original train.py.

Saves the EMA weights in the exact same checkpoint format as your original
best.pt, so sample.py can load it with NO changes -- just point --weightfile
at the new ema_best.pt.

Usage (pilot test on fibrosis first, since it showed the worst mode collapse):
  python train_ema_finetune.py --class_idx 3 \
      --weightfile outputs/fibrosis/best.pt \
      --output_dir outputs/fibrosis_ema \
      --extra_epochs 300
"""

import os
import copy
import argparse
import torch
from torch.optim import Adam

from ddpm import UNet, DDPM
from dataset import get_dataloader, CLASS_NAMES


class EMA:
    """Maintains a shadow copy of model parameters, updated as:
    shadow = decay * shadow + (1 - decay) * param
    after every optimizer step."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())

    @torch.no_grad()
    def update(self, model):
        model_sd = model.state_dict()
        for name, shadow_val in self.shadow.items():
            if shadow_val.dtype.is_floating_point:
                self.shadow[name] = self.decay * shadow_val + (1 - self.decay) * model_sd[name]
            else:
                # non-float buffers (e.g. registered int buffers) -- just copy
                self.shadow[name] = model_sd[name].clone()

    def state_dict(self):
        return self.shadow


def finetune(class_idx, npy_dir, weightfile, output_dir, extra_epochs,
             batch_size, lr, timesteps, ema_decay, save_every):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Resuming class: {CLASS_NAMES[class_idx]} (idx={class_idx})")
    print(f"Loading from: {weightfile}")

    # ── Data ─────────────────────────────────────────────────────────────────
    loader = get_dataloader(npy_dir, class_idx, batch_size=batch_size)

    # ── Model: load existing trained weights ────────────────────────────────
    unet = UNet(channels=64, time_dim=256).to(device)
    ddpm = DDPM(unet, timesteps=timesteps).to(device)
    ckpt = torch.load(weightfile, map_location=device, weights_only=False)
    ddpm.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (loss={ckpt['loss']:.4f})")

    # ── EMA: initialize shadow weights from the current (already-trained) model ──
    ema = EMA(ddpm, decay=ema_decay)

    # ── Optimiser: small LR for fine-tuning, not the original training LR ──────
    # (the model is already trained; we don't want to disturb it much, just
    # let EMA smooth out noise over a modest number of additional steps)
    opt = Adam(ddpm.parameters(), lr=lr)

    # ── Output dir ────────────────────────────────────────────────────────────
    class_name = CLASS_NAMES[class_idx]
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fine-tuning for {extra_epochs} additional epochs, lr={lr}, ema_decay={ema_decay}")

    for epoch in range(1, extra_epochs + 1):
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
            ema.update(ddpm)  # update EMA shadow after every step
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{extra_epochs} | loss: {avg_loss:.4f}")

        if epoch % save_every == 0 or epoch == extra_epochs:
            # Save EMA weights in the SAME format as the original best.pt,
            # so sample.py works with zero changes.
            torch.save({
                'epoch': ckpt['epoch'] + epoch,
                'model_state_dict': ema.state_dict(),
                'loss': avg_loss,
                'class_idx': class_idx,
                'class_name': class_name,
                'note': f'EMA weights, decay={ema_decay}, fine-tuned {epoch} extra epochs from {weightfile}',
            }, os.path.join(output_dir, f'ema_epoch_{epoch:04d}.pt'))

    # Final EMA checkpoint, named to match what sample.py expects by default
    torch.save({
        'epoch': ckpt['epoch'] + extra_epochs,
        'model_state_dict': ema.state_dict(),
        'loss': avg_loss,
        'class_idx': class_idx,
        'class_name': class_name,
        'note': f'EMA weights, decay={ema_decay}, fine-tuned {extra_epochs} extra epochs from {weightfile}',
    }, os.path.join(output_dir, 'ema_best.pt'))

    print(f"\nDone. EMA checkpoint saved to: {os.path.join(output_dir, 'ema_best.pt')}")
    print(f"Sample from it with the existing sample.py, unchanged:")
    print(f"  python sample.py --class_idx {class_idx} --weightfile {os.path.join(output_dir, 'ema_best.pt')} "
          f"--n_samples 200 --output_dir outputs/synthetic_test_ema")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_idx',  type=int,   required=True,
                        help='Class index (0=healthy,1=emphysema,2=ground_glass,3=fibrosis,4=micronodules)')
    parser.add_argument('--weightfile', type=str,   required=True,
                        help='Path to existing trained checkpoint (e.g. outputs/fibrosis/best.pt)')
    parser.add_argument('--npy_dir',    type=str,   default='/rds/general/user/eh1121/home/Final_Project/ILD_DB_npy')
    parser.add_argument('--output_dir', type=str,   required=True)
    parser.add_argument('--extra_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=2e-5,
                        help='Small LR for fine-tuning -- much lower than original training LR (2e-4)')
    parser.add_argument('--timesteps',  type=int,   default=1000)
    parser.add_argument('--ema_decay',  type=float, default=0.999)
    parser.add_argument('--save_every', type=int,   default=100)
    args = parser.parse_args()

    finetune(
        class_idx=args.class_idx,
        npy_dir=args.npy_dir,
        weightfile=args.weightfile,
        output_dir=args.output_dir,
        extra_epochs=args.extra_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        ema_decay=args.ema_decay,
        save_every=args.save_every,
    )