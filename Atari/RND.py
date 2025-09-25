import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNDModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128, device="cpu"):
        super().__init__()
        self.device = device

        # Frozen random target net
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)
        for p in self.target.parameters():
            p.requires_grad = False

        # Trainable predictor net
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

    def forward(self, x):
        with torch.no_grad():
            target_out = self.target(x)
        pred_out = self.predictor(x)
        return pred_out, target_out

    def intrinsic_reward(self, x):
        pred, target = self.forward(x)
        loss = F.mse_loss(pred, target, reduction="none")
        return loss.mean(dim=1) / loss.std(dim=1)