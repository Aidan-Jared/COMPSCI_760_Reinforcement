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
        all_rewards = rewards.flatten()
        batch_mean = all_rewards.mean().item()
        batch_var = all_rewards.var(unbiased=False).item()
        tot_count = all_rewards.shape[0] + self.count
        self.running_mean = self.running_mean + (batch_mean - self.running_mean) * all_rewards.shape[0] / tot_count
        self.running_var  = self.running_var + (batch_var - self.running_var) * all_rewards.shape[0] / tot_count

    def intrinsic_reward(self, x, normalize=False):
        pred, target = self.forward(x)
        # per-sample squared error
        loss = F.mse_loss(pred, target, reduction="none")

        if normalize:
            if self.count < 1000:
                self.update_rms(loss.detach())
                self.count += 1
            self.update_rms(loss.detach())


            std = torch.clamp(self.running_var.sqrt(), min=0.1)  # Lower minimum
            normalized = (loss - self.running_mean.to(self.device)) / std.to(self.device)
            return torch.clamp(normalized.mean(), 0, 5)
        else:
            return torch.clamp(loss.mean() / 10, 0, 100)
