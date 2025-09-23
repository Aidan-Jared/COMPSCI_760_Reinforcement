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
        self.manager = ManagerTransformer(self.d, self.c, self.r, args, device, n_actions)
        self.worker = Worker(self.b, self.c, self.d, self.k, n_actions, device, args)


        self.hidden_m = self._init_hidden(args.num_workers, self.r * self.d, grad=True)
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

    def forward(self, x, zs, actions, rewards, goals, states, frames, mask, save=True):
        x = self.preprocessor(x)
        z = self.perception(x)
        if torch.isnan(z).any():
            print('here')
        if len(zs) >= (2 * self.c + 1):
            zs.pop(0)
        zs.append(z)

        goal, state, value_m, hidden_m = self.manager(zs, actions, rewards, goals, frames, self.hidden_m, mask) #
        
        if len(goals) >= (2 * self.c + 1):
            goals.pop(0)
            states.pop(0)
        
        goals.append(goal)
        states.append(state.detach())

        action_dist, hidden_w, value_w = self.worker(z,  goals[:self.c + 1], self.hidden_w, mask[-1])
        if save:
            self.hidden_w = (hidden_w[0].detach(),hidden_w[1].detach())
            self.hidden_m = (hidden_m[0].detach(),hidden_m[1].detach())

        return action_dist, goals, states, zs, value_m, value_w
    
    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)
    
    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)
    
    def repackage_hidden(self):
        def repackage_rnn(x):
            return[item.detach() for item in x]
        self.hidden_w = repackage_rnn(self.hidden_w)
        self.hidden_m = repackage_rnn(self.hidden_m)
    
    def init_obj(self):
        template = torch.zeros(self.b, self.d)
        goals = [torch.zeros_like(template).to(self.device) for _ in range(2*self.c+1)]
        states = [torch.zeros_like(template).to(self.device) for _ in range(2*self.c+1)]
        masks = [torch.ones(self.b, 1).to(self.device) for _ in range(2*self.c+1)]
        actions = [torch.zeros((self.b,1)).to(self.device) for _ in range(2*self.c+1)]
        rewards = [torch.zeros((self.b,1)).to(self.device) for _ in range(2*self.c+1)]
        zs = [torch.zeros((self.b,self.d)).to(self.device) for _ in range(2*self.c)]
        frames = [torch.zeros((self.b,1)).to(self.device) for _ in range(2*self.c+1)]
        return goals, states, masks, actions, rewards, zs, frames
    
    def eps_decay(self):
        self.manager.eps *= self.decay
        # self.worker.eps *= self.decay
    
    def detach_sequences(self, goals, states, zs, actions, rewards, frames):
        """Detach all sequences to prevent backprop through previous steps"""
        goals = [g.detach() for g in goals]
        states = [s.detach() for s in states] 
        zs = [z.detach() for z in zs]  # This is crucial!
        actions = [a.detach() for a in actions]  # Detach action history
        rewards = [r.detach() for r in rewards]  # Detach reward history
        frames = [f.detach() for f in frames]  # Detach reward history
        return goals, states, zs, actions, rewards, frames
    
class GRUGate(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.reset_x = nn.Linear(d_model, d_model, bias=False)
        self.reset_y = nn.Linear(d_model, d_model, bias=False)
        self.update_x = nn.Linear(d_model, d_model, bias=False)
        self.update_y = nn.Linear(d_model, d_model, bias=False)
        self.update_bias = nn.Parameter(torch.ones(d_model)*.1)
        self.canidate_x = nn.Linear(d_model, d_model, bias=False)
        self.canidate_y = nn.Linear(d_model, d_model, bias=False)

        self._init_conservative()

    def _init_conservative(self):
        for module in [self.reset_x, self.reset_y, self.canidate_x, self.canidate_y]:
            nn.init.xavier_uniform_(module.weight, gain=.3)

            
        for module in [self.update_x, self.update_y]:
            nn.init.xavier_uniform_(module.weight, gain=.5)
    
    def forward(self,x,y):
        reset = torch.sigmoid(self.reset_x(x) + self.reset_y(y))
        candidate = torch.tanh(
            self.canidate_y(y) + self.canidate_x(reset * x)
        )
        update = torch.sigmoid(
            self.update_x(x) + self.update_y(y) - self.update_bias
        )
        return (1 - update) * x + update * candidate
        

class GattedTransformerEncoderLayer(nn.Module):
    def __init__(self, d, nhead, dropout = .1):
        super().__init__()
        self.d = d
        self.self_attn = nn.MultiheadAttention(embed_dim=self.d, num_heads=nhead, dropout=dropout)
        self.linear1 = nn.Linear(self.d, self.d)
        self.linear2 = nn.Linear(self.d, self.d)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)

        self.gate1 = GRUGate(self.d)
        self.gate2 = GRUGate(self.d)
        self.eps = .01

    def forward(self, src, src_mask=None, src_key_padding_mask = None, is_causal=None):
        
        src_norm = self.norm1(src)
        attn_output, _ = self.self_attn(src_norm, src_norm, src_norm, attn_mask = src_mask, key_padding_mask = src_key_padding_mask)
        # if self.training and np.random.rand() < self.eps:
        #     return attn_output

        src = self.gate1(src, attn_output)

        src_norm = self.norm2(src)

        ffn_output = self.dropout2(self.linear2(self.dropout1(F.gelu(self.linear1(src_norm)))))

        src = self.gate2(src, ffn_output)

        return src

class ManagerTransformer(nn.Module):
    def __init__(self, d, k, dialation, args, device, n_actions):
        super().__init__()
        self.d = d
        self.k = k
        self.r = dialation
        self.eps = args.eps
        self.device = device
        self.action_embed = nn.Linear(1, self.d)
        # self.action_embed = nn.Embedding(n_actions, self.d)
        self.reward_embed = nn.Linear(1, self.d, bias=False)
        self.Mspace = nn.Linear(self.d, self.d)
        self.time_embed = nn.Linear(1, self.d, bias=False)
        self.index = torch.arange(0, self.r * self.d, self.r)
        self.dilation = 0

        # position = torch.arange(self.k + 1).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.d, 2) * (-math.log(10000.0) / self.d))
        # self.pe = torch.zeros(self.k + 1, 1, self.d).to(device)
        # self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        # self.pe[:, 0, 1::2] = torch.cos(position * div_term)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=4, dim_feedforward=128, dropout=0, batch_first=False)
        # self.transformer = nn.Transformer(d_model=self.d, nhead=4, dim_feedforward=self.d, dropout= 0, batch_first=False, num_decoder_layers=1, num_encoder_layers=1)
        encoder_layer =  GattedTransformerEncoderLayer(self.d, 4)
        self.encoders = nn.TransformerEncoder(encoder_layer, 1)

        self.goal_gate = GRUGate(self.d)

        self.goal_transform = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.Tanh(),
            nn.Linear(self.d, self.d)
        )

        # self.goal_transform = nn.Sequential(
        #     nn.Linear(self.d, self.d),
        #     nn.GELU(),
        #     nn.Linear(self.d, self.d)
        # )


        # self.fc = nn.Linear(self.d, self.d)
        self.critic = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, self.d // 2),
            nn.GELU(),
            nn.Linear(self.d //2, 1)
        )

        self.memory_update = nn.Linear(self.d * 2, self.d)

        self._stable_init()

    def _stable_init(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    
    @property
    def dilation_idx(self):
        dilation_idx = self.dilation + self.index
        self.dilation = (self.dilation + 1) % self.r
        return dilation_idx
    
    def masked_idx(self, dilated_idx):
        all_indices = torch.arange(self.r * self.d)
        mask = torch.ones(self.r * self.d, dtype=torch.bool)
        mask[dilated_idx] = False
        masked_idx = all_indices[mask]
        return masked_idx

    def forward(self, zs, actions, rewards, goals, frames, hidden, mask):
        zs = torch.stack(zs)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        goals = torch.stack(goals)
        frames = torch.stack(frames)
        # if torch.sum(mask[-1]) < 8:
        #     print('here')
        mask = torch.stack(mask)
        states = F.gelu(self.Mspace(zs))
        state = states[-1]
        a_embed = self.action_embed(actions)
        r_embed = self.reward_embed(rewards)
        frames_embed = self.time_embed(frames)
        
        # sequence = []
        # for t in range(self.k + 1):
        #     sequence.append(((states[t]) * mask[t]) + frames_embed[t])
        #     sequence.append(((goals[t]) * mask[t]) + frames_embed[t])
        #     sequence.append(((a_embed[t]) * mask[t]) + frames_embed[t])
        #     sequence.append(((r_embed[t]) * mask[t]) + frames_embed[t])
        # sequence = torch.stack(sequence)
        d_idx = self.dilation_idx
        recent_history = torch.cat(((states[-self.k:] + frames_embed[-self.k:]) * mask[-self.k:], (goals[-self.k:] + frames_embed[-self.k:]) * mask[-self.k:],  (a_embed[-self.k:] + frames_embed[-self.k:] + r_embed[-self.k:]) * mask[-self.k:]), dim=0)
        long_view = torch.cat(((states[:-self.k:3] + frames_embed[:-self.k:3]) * mask[:-self.k:3], (goals[:-self.k:3] + frames_embed[:-self.k:3]) * mask[:-self.k:3],  (a_embed[:-self.k:3] + frames_embed[:-self.k:3] + r_embed[:-self.k:3]) * mask[:-self.k:3]), dim=0)

        hx, cx = hidden
        
        sequence= torch.cat((recent_history, long_view, hx[:,d_idx].unsqueeze(0) * mask[-1], cx[:,d_idx].unsqueeze(0) * mask[-1]), dim=0) #(r_embed  * mask) + frames_embed

        # sequence = torch.clamp(sequence, -3, 3)
        
        goal_hat = self.encoders(sequence)[-1]
        cx[:,d_idx] = goal_hat

        combined = torch.cat([goal_hat, hx[:,d_idx]],dim=-1)

        hx[:,d_idx] = torch.tanh(self.memory_update(combined))

        detatached_hx = hx[:, self.masked_idx(d_idx)].detach()
        detatached_hx = detatached_hx.view(detatached_hx.shape[0], self.d, self.r - 1)
        detatached_hx = detatached_hx.sum(-1)

        goal_hat = (goal_hat + detatached_hx) / self.r

        
        # goal_hat = self.transformer(sequence, sequence[-1].unsqueeze(0)).squeeze(0)
        # goal_hat = goal_hat.reshape([goal_hat.shape[1], goal_hat.shape[0] * goal_hat.shape[2]])

        goal_raw = torch.tanh(self.goal_transform(goal_hat))
        # goal_raw = self.goal_gate(goal_hat, goal_transform)

        # goal_norm = torch.norm(goal_transform, dim=-1, keepdim=True)
        # goal_norm = torch.clamp(goal_norm, min=1e-6, max=10)

        # goal = goal_transform / goal_norm4

        if self.training and np.random.rand() < self.eps:
            noise = torch.rand_like(goal_raw) * .01
            goal_raw = goal_raw + noise

        
        goal = F.normalize(goal_raw, dim=-1, eps=1e-6)
        
        value_est = self.critic(goal_hat)
        value_est = torch.clamp(value_est, -50, 50)
        # goal_hat = self.fc(goal_hat[-1])
        
        state = state.detach()
        return goal, state, value_est, hidden
    
    def state_goal_cosine(self, states, goals, masks):

        t = self.k
        mask = torch.stack(masks[t: t+ self.k]).prod(dim=0)
        cos_d = F.cosine_similarity(states[t + self.k] - states[t], goals[t])
        cos_d = mask * cos_d.unsqueeze(-1)
        return cos_d


        # t = self.k
        # mask = torch.stack(masks[t+1: t+ self.k])#.prod(dim=0)
        # compair = torch.linspace(.03,1,self.k-1).unsqueeze(-1).repeat(1,mask.shape[1]).to(self.device)
        # cos_d = F.cosine_similarity((torch.stack(states[t:t + self.k - 1]) - states[t + self.k]), goals[t], dim = -1)
        # cos_d = -(compair - (cos_d * mask.squeeze(2))).mean(0).unsqueeze(-1)
        # # full_look = torch.sum(mask, dim=0) / 30 
        # # full_look[full_look < 1]  /= 1
        # return cos_d #* full_look.to(self.device)


if __name__ == "__main__":
    test = ManagerTransformer(100,10,1,"test", "cpu")
    x = torch.rand((10,100))
    actions = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype=torch.float32)
    rewards = torch.tensor([[0],[0],[0],[0],[1],[0],[-1],[0],[1],[1]], dtype=torch.float32)
    test(x, actions, rewards, 'test')