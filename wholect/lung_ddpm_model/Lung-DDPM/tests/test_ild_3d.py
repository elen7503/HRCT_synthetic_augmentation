from __future__ import annotations

import unittest

import torch

from ild_3d import GaussianDiffusion3D, NUM_CONDITION_CHANNELS_3D


class ZeroModel(torch.nn.Module):
    def forward(self, value, timestep):
        return torch.zeros_like(value[:, :1])


class Diffusion3DTests(unittest.TestCase):
    def test_weighted_loss_and_sampling_shapes(self):
        diffusion = GaussianDiffusion3D(ZeroModel(), volume_size=8, timesteps=3)
        clean = torch.zeros((1, 1, 8, 8, 8))
        condition = torch.zeros((1, NUM_CONDITION_CHANNELS_3D, 8, 8, 8))
        condition[:, 0, 1:7, 1:7, 1:7] = 1
        condition[:, 4, 3:5, 3:5, 3:5] = 1
        loss = diffusion(clean, condition)
        self.assertTrue(torch.isfinite(loss))
        generated = diffusion.sample(condition, clean, condition[:, :1], strength=1, guidance_scale=1)
        self.assertEqual(generated.shape, clean.shape)
        self.assertTrue(torch.allclose(generated[:, :, 0, 0, 0], clean[:, :, 0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
