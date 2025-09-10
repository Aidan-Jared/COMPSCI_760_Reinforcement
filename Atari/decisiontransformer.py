import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ManagerTransformer(nn.Module):
    def __init__(self, d, k, layers, args, n_actions, device):
        super().__init__()
        self.d = d
        self.k = k
        self.layers = layers
        self.eps = args.eps
        self.n_actions = n_actions
        self.device = device
        self.action_embed = nn.Linear(1, self.d)
        self.reward_embed = nn.Linear(1, self.d)
        self.Mspace = nn.Linear(self.d, self.d)

        layer = nn.TransformerEncoderLayer(d_model=self.d * 3, nhead=8, dim_feedforward=512, dropout=0.1, batch_first=True)
        self.encoders = nn.TransformerEncoder(layer, 4)

        self.fc = nn.Linear(self.d * (self.k) * 3, self.n_actions)
        self.critic = nn.Linear(self.d * (self.k) * 3, 1)
    
    def forward(self, zs, actions, rewards, goals, mask):
        zs = torch.stack(zs)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        mask = torch.stack(mask)
        states = F.relu(self.Mspace(zs * mask))
        state = states[-1]
        a_embed = self.action_embed(actions * mask)
        r_embed = self.reward_embed(rewards * mask)
        chrono_emb= torch.cat((states, a_embed, r_embed), dim=2)
        
        a_hat = self.encoders(chrono_emb)
        a_hat = a_hat.reshape([a_hat.shape[1], a_hat.shape[0] * a_hat.shape[2]])

        value_est = self.critic(a_hat)
        actions = self.fc(a_hat)
        
        state = state.detach()
        actions = F.normalize(actions, dim=-1, eps=1e-6)
        # if (self.eps > torch.rand(1)[0]):
        #     goal = torch.randn_like(goal, requires_grad=False)
        return actions, state, value_est
