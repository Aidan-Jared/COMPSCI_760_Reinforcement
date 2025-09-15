import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from feudalnet import Perception, Preprocessor, Worker

class FeudalTransformer(nn.Module):
    def __init__(self, num_workers, input_dim, hidden_dim_manager, hidden_dim_worker, n_actions, time_horizon = 10, dilation = 10, device = 'cpu', mlp = False, args = None):
        super().__init__()
        self.b = num_workers
        self.c = time_horizon
        self.d = hidden_dim_manager
        self.k = hidden_dim_worker
        self.r = dilation
        self.n_actions = n_actions
        self.device = device
        self.decay = args.decay

        self.preprocessor = Preprocessor(input_dim, device, mlp)

        self.perception = Perception(input_dim, self.d, mlp)
        self.manager = ManagerTransformer(self.d, self.c, self.r, args, device)
        self.worker = Worker(self.b, self.c, self.d, self.k, n_actions, device, args)


        # self.hidden_m = self._init_hidden(args.num_workers, self.r * self.d, grad=True)
        self.hidden_w = self._init_hidden(args.num_workers, self.k * n_actions, grad=True)
        
        self.args = args
        self.to(device)
        self.apply(self._weight_init)

    def _init_hidden(self, n_workers, h_dim, grad=False):
        return(torch.zeros(n_workers, h_dim, requires_grad=grad).to(self.device), torch.zeros(n_workers, h_dim, requires_grad=grad).to(self.device))

    def _weight_init(self, layer):
        if type(layer) == nn.modules.conv.Conv2d or type(layer) == nn.Linear:
            nn.init.orthogonal_(layer.weight.data)
            if layer.bias is not None:
                nn.init.constant_(layer.bias.data,0)

    def forward(self, x, zs, actions, rewards, goals, states, mask, save=True):
        x = self.preprocessor(x)
        z = self.perception(x)
        if torch.isnan(z).any():
            print('here')
        if len(zs) >= (2 * self.c + 1):
            zs.pop(0)
        zs.append(z)

        goal, state, value_m = self.manager(zs, actions, rewards, goals, mask)
        
        if len(goals) >= (2 * self.c + 1):
            goals.pop(0)
            states.pop(0)
        
        goals.append(goal)
        states.append(state.detach())

        action_dist, hidden_w, value_w = self.worker(z,  goals[:self.c + 1], self.hidden_w, mask[-1])
        if save:
            self.hidden_w = (hidden_w[0].detach(),hidden_w[1].detach())

        return action_dist, goals, states, zs, value_m, value_w
    
    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)
    
    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)
    
    def repackage_hidden(self):
        def repackage_rnn(x):
            return[item.detach() for item in x]
        self.hidden_w = repackage_rnn(self.hidden_w)
    
    def init_obj(self):
        template = torch.zeros(self.b, self.d)
        goals = [torch.zeros_like(template).to(self.device) for _ in range(2*self.c+1)]
        states = [torch.zeros_like(template).to(self.device) for _ in range(2*self.c+1)]
        masks = [torch.ones(self.b, 1).to(self.device) for _ in range(2*self.c+1)]
        actions = [torch.zeros((self.b,1)).to(self.device) for _ in range(2*self.c+1)]
        rewards = [torch.zeros((self.b,1)).to(self.device) for _ in range(2*self.c+1)]
        zs = [torch.zeros((self.b,self.d)).to(self.device) for _ in range(2*self.c)]
        return goals, states, masks, actions, rewards, zs
    
    def eps_decay(self):
        self.manager.eps *= self.decay
        self.worker.eps *= self.decay
    
    def detach_sequences(self, goals, states, zs, actions, rewards):
        """Detach all sequences to prevent backprop through previous steps"""
        goals = [g.detach() for g in goals]
        states = [s.detach() for s in states] 
        zs = [z.detach() for z in zs]  # This is crucial!
        actions = [a.detach() for a in actions]  # Detach action history
        rewards = [r.detach() for r in rewards]  # Detach reward history
        return goals, states, zs, actions, rewards


class ManagerTransformer(nn.Module):
    def __init__(self, d, k, layers, args, device):
        super().__init__()
        self.d = d
        self.k = k
        self.layers = layers
        self.eps = args.eps
        self.device = device
        self.action_embed = nn.Linear(1, self.d)
        self.reward_embed = nn.Linear(1, self.d)
        self.Mspace = nn.Linear(self.d, self.d)

        position = torch.arange((self.k * 2) + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d, 2) * (-math.log(10000.0) / self.d))
        self.pe = torch.zeros((self.k * 2) + 1, 1, self.d).to(device)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

        layer = nn.TransformerEncoderLayer(d_model=self.d * 4, nhead=4, dim_feedforward=512, dropout=0, batch_first=False)
        self.encoders = nn.TransformerEncoder(layer, 2)

        self.fc = nn.Linear(self.d * (self.k * 2 + 1) * 4, self.d)
        self.critic = nn.Linear(self.d * (self.k * 2 + 1) * 4, 1)
    
    def forward(self, zs, actions, rewards, goals, mask):
        zs = torch.stack(zs)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        goals = torch.stack(goals)
        mask = torch.stack(mask)
        states = F.relu(self.Mspace(zs * mask))
        state = states[-1]
        a_embed = self.action_embed(actions * mask)
        r_embed = self.reward_embed(rewards * mask)
        chrono_emb= torch.cat(((states + self.pe) * mask, (goals  + self.pe) * mask, (a_embed  + self.pe) * mask, (r_embed  + self.pe) * mask), dim=2)
        
        goal_hat = self.encoders(chrono_emb)
        goal_hat = goal_hat.reshape([goal_hat.shape[1], goal_hat.shape[0] * goal_hat.shape[2]])

        value_est = self.critic(goal_hat)
        goal_hat = self.fc(goal_hat)
        
        state = state.detach()
        goal = F.normalize(goal_hat, dim=-1, eps=1e-6)
        return goal, state, value_est
    
    def state_goal_cosine(self, states, goals, masks):
        t = self.k
        mask = torch.stack(masks[t: t+ self.k]).prod(dim=0)
        cos_d = F.cosine_similarity(states[t + self.k] - states[t], goals[t])
        cos_d = mask * cos_d.unsqueeze(-1)
        return cos_d


if __name__ == "__main__":
    test = ManagerTransformer(100,10,1,"test", "cpu")
    x = torch.rand((10,100))
    actions = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype=torch.float32)
    rewards = torch.tensor([[0],[0],[0],[0],[1],[0],[-1],[0],[1],[1]], dtype=torch.float32)
    test(x, actions, rewards, 'test')