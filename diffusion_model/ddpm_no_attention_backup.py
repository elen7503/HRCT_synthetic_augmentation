"""
Simple 2D DDPM with U-Net for 32x32 HRCT patches.
One model trained per tissue class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        embeddings = math.log(10000) / (half - 1)
        embeddings = torch.exp(torch.arange(half, device=device) * -embeddings)
        embeddings = t[:, None].float() * embeddings[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.GroupNorm(1 if in_ch < 8 else 8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual(x)


class UNet(nn.Module):
    def __init__(self, channels=64, time_dim=256):
        super().__init__()
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.enc1 = ResBlock(1,          channels,     time_dim)
        self.enc2 = ResBlock(channels,   channels * 2, time_dim)
        self.enc3 = ResBlock(channels*2, channels * 4, time_dim)

        self.down1 = nn.Conv2d(channels,   channels,   4, 2, 1)
        self.down2 = nn.Conv2d(channels*2, channels*2, 4, 2, 1)

        self.mid = ResBlock(channels * 4, channels * 4, time_dim)

        self.up1  = nn.ConvTranspose2d(channels*4, channels*2, 4, 2, 1)
        self.dec1 = ResBlock(channels*4, channels*2, time_dim)

        self.up2  = nn.ConvTranspose2d(channels*2, channels, 4, 2, 1)
        self.dec2 = ResBlock(channels*2, channels,  time_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        m  = self.mid(e3, t_emb)
        d1 = self.dec1(torch.cat([self.up1(m), e2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1), t_emb)
        return self.out(d2)


class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.T = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', alphas_cumprod.sqrt())
        self.register_buffer('sqrt_one_minus_alphas_cumprod', (1 - alphas_cumprod).sqrt())

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sqrt_alphas_cumprod[t][:, None, None, None]
        b = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return a * x0 + b * noise, noise

    def loss(self, x0):
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        noisy, noise = self.q_sample(x0, t)
        pred = self.model(noisy, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n, device, img_size=32):
        x = torch.randn(n, 1, img_size, img_size, device=device)
        for t in reversed(range(self.T)):
            t_batch = torch.full((n,), t, device=device, dtype=torch.long)
            pred_noise = self.model(x, t_batch)
            beta = self.betas[t]
            alpha = 1 - beta
            alpha_bar = self.alphas_cumprod[t]
            x = (1 / alpha.sqrt()) * (
                x - (beta / (1 - alpha_bar).sqrt()) * pred_noise
            )
            if t > 0:
                alpha_bar_prev = self.alphas_cumprod[t - 1]
                posterior_var = (1 - alpha_bar_prev) / (1 - alpha_bar) * beta
                x = x + posterior_var.sqrt() * torch.randn_like(x)
        return x
