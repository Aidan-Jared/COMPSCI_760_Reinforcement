import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections  import deque

class RNDModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128, device="cpu"):
        super().__init__()
        self.device = device

        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

        # keep track of mean/std for normalization
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.count = 1e-4  # to avoid div/0

    def forward(self, x):
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        return pred_out, target_out

    def update_rms(self, rewards):
        # simple running mean/std (could also use Welford’s algorithm)
        batch_mean = rewards.mean().item()
        batch_var = rewards.var(unbiased=False).item()
        m, v, n = self.running_mean.item(), self.running_var.item(), self.count
        new_n = n + 1
        new_m = m + (batch_mean - m) / new_n
        new_v = v + (batch_var - v) / new_n
        self.running_mean.fill_(new_m)
        self.running_var.fill_(new_v)
        self.count = new_n
        self.running_mean

    def intrinsic_reward(self, x, normalize=False):
        pred, target = self.forward(x)
        # per-sample squared error
        loss = F.mse_loss(pred, target, reduction="none").mean(dim=1)

        if normalize:
            self.update_rms(loss.detach())
            std = torch.clamp(self.running_var.sqrt(), min = 1.).to(self.device)
            mean = self.running_mean.to(self.device)
            normalized = (loss - mean) / std
            return torch.clamp(normalized, 0, 1)
        else:
            return torch.clamp(loss / 10, 0, 1)
