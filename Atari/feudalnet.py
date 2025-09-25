import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dilated_lstm import DilatedLSTM
from utils import VectorEnvVisualizer, Storage

class FeudalNetwork(nn.Module):
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
        self.manager = Manager(self.c, self.d, self.r, args, device)
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

    def forward(self, x, goals, states, mask, save=True):
        x = self.preprocessor(x)
        z = self.perception(x)
        goal, hidden_m, state, value_m = self.manager(z, self.hidden_m, mask)

        if len(goals) > (2 * self.c + 1):
            goals.pop(0)
            states.pop(0)
        
        goals.append(goal)
        states.append(state.detach())
        
        action_dist, hidden_w, value_w = self.worker(z,  goals[:self.c + 1], self.hidden_w, mask)
        if save:
            self.hidden_m = (hidden_m[0].detach(),hidden_m[1].detach())
            self.hidden_w = (hidden_w[0].detach(),hidden_w[1].detach())
        return action_dist, goals, states, value_m, value_w
    
    def intrinsic_reward(self, states, goals, masks):
        return self.worker.intrinsic_reward(states, goals, masks)
    
    def state_goal_cosine(self, states, goals, masks):
        return self.manager.state_goal_cosine(states, goals, masks)
    
    def goal_entropy(self, goals, masks):
        return self.manager.goal_entropy(goals, masks)
    
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
        return goals, states, masks
    
    def eps_decay(self):
        try:
            self.manager.eps *= self.decay
            self.worker.eps *= self.decay
        except:
            self.worker.eps *= self.decay
class Perception(nn.Module):
    def __init__(self, input_dim, d, mlp = False):
        super().__init__()
        if mlp:
            self.percept = nn.Sequential(
                nn.Linear(input_dim[-1], 64),
                nn.LayerNorm(64),
                nn.LeakyReLU(.01),
                nn.Linear(64, d),
                nn.LeakyReLU(.01),)
        else:
            w1 = (input_dim[0] - 8) / 4 + 1
            h1 = (input_dim[1] - 8) / 4 + 1
            w2 = int((w1 - 4) / 2 + 1)
            h2 = int((h1 - 4) / 2 + 1)
            self.percept = nn.Sequential(
                nn.Conv2d(3,16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(16,32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.modules.Flatten(),
                nn.Linear(32*w2 * h2, d),
                nn.ReLU()
            )
    
    def forward(self, x):
        return self.percept(x)
        # return torch.utils.checkpoint.checkpoint(self.percept, x, use_reentrant=False)

class Manager(nn.Module):
    def __init__(self, c, d, r, args, device):
        super().__init__()
        self.c = c # time horizon
        self.d = d # hidden dim size
        self.r = r # dilation elvel
        self.device = device

        self.Mspace = nn.Linear(self.d, self.d)
        self.Mrnn = DilatedLSTM(self.d, self.d, self.r)
        self.critic = nn.Linear(self.d, 1)

    def forward(self, z, hidden, mask):
        state = F.relu(self.Mspace(z))
        hidden = (mask * hidden[0], mask * hidden[1])
        goal_hat, hidden = self.Mrnn(state, hidden)
        value_est = self.critic(goal_hat)

        goal = F.normalize(goal_hat)
        state = state.detach()

        return goal, hidden, state, value_est
    
    def state_goal_cosine(self, states, goals, masks):
        '''
        the cosine similarity between the the last c timesteps and the current state to the managers goal
        '''
        t = self.c
        mask = torch.stack(masks[t: t+ self.c]).prod(dim=0)
        cos_d = F.cosine_similarity(states[t + self.c] - states[t], goals[t])
        cos_d = mask * cos_d.unsqueeze(-1)
        return cos_d
    
    def goal_entropy(self,goals, masks):
        t = self.c
        current_goal = goals[self.c]
        mask = torch.stack(masks[t: t+ self.c]).prod(dim=0).squeeze(-1)

        masked_goal = current_goal[mask.bool()]
        
        similarities = torch.mm(masked_goal, masked_goal.t())

        distances = 1 - similarities

        distances = distances[~torch.eye(distances.shape[0], dtype=torch.bool, device=self.device)]


        mean_distance = distances.mean()
        var_distance = distances.var() + 1e-6

        entropy = 0.5 * torch.log(2 * torch.pi * var_distance) + mean_distance

        return entropy
    
class Worker(nn.Module):
    def __init__(self, b, c, d, k, num_actions, device, args):
        super().__init__()
        self.b = b
        self.c = c
        self.k = k
        self.num_actions = num_actions
        self.device = device
        self.eps = args.eps

        self.Wrnn = nn.LSTMCell(d, k * self.num_actions)
        self.phi = nn.Linear(d, k, bias=False)

        self.critic = nn.Sequential(
            nn.Linear(k * num_actions, 50),
            nn.ReLU(),
            nn.Linear(50,1)
        )
    
    def forward(self, z, goals, hidden, mask):
        hidden = (mask * hidden[0], mask * hidden[1])
        u, cx = self.Wrnn(z, hidden)
        hidden = (u, cx)

        goals = torch.stack(goals).detach().sum(dim=0)
        # weights = torch.exp(-0.02 * torch.arange(len(goals), dtype=torch.float, device=z.device))
        # weights = weights / weights.sum()

        # weighted_goals = torch.einsum('t,tbd->bd', weights.flip(0), goals)
        w = self.phi(goals)
        value_est = self.critic(u)

        u = u.reshape(u.shape[0], self.k, self.num_actions)
        a = F.softmax(torch.einsum("bk, bka -> ba", w, u), dim=-1)

        if self.training and np.random.rand() < self.eps:
            noise = torch.rand_like(a) * .1
            a = a + noise

        return a, hidden, value_est
    
    def intrinsic_reward(self, states, goals, masks):
        t = self.c
        r_i = torch.zeros(self.b, 1).to(self.device)
        mask = torch.ones(self.b, 1).to(self.device)
        for i in range(1, self.c + 1):
            r_i_t = F.cosine_similarity(states[t] - states[t - i], goals[t - i]).unsqueeze(-1)
            r_i += (mask * r_i_t)

            mask = mask * masks[t - i]
        r_i = r_i.detach()
        return r_i / self.c
    
class Preprocessor:
    def __init__(self, shape, device='cpu', mlp=False):
        self.mlp = mlp
        if mlp:
            self.shape = (shape[-1],)
        else:
            self.shape = (shape[-1], shape[0], shape[1])
        self.device = device
        self.rms = RunningMeanStd(shape = (1,) + self.shape)

    def __call__(self, x):
        if not self.mlp:
            x = np.asarray(x).reshape(x.shape[0], *self.shape)
            self.rms.update(x)
            
            # Check if std is reasonable
            std = np.sqrt(self.rms.var + 1e-5)
            if std.mean() < 1e-3:  # Too small, use simple normalization
                x_normalized = (x - 128.0) / 64.0  # RAM values [0,255] -> ~[-2,2]
            else:
                x_normalized = (x - self.rms.mean) / std
            
            # CRITICAL: Clip to reasonable range
            x_normalized = np.clip(x_normalized, -3.0, 3.0)
            
            return torch.FloatTensor(x_normalized).to(self.device)
        else:
            return torch.FloatTensor(x).to(self.device)

class Qlearn(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, device, mlp):
        super().__init__()
        self.d = int(hidden_dim)
        self.n_actions = int(n_actions)
        self.device = device

        self.preprocessor = Preprocessor(input_dim, device, mlp)
        self.perception   = Perception(input_dim, self.d, mlp)  # outputs [B, d]

        # Robust integer hidden sizes (avoid float dims from /2, /4)
        h1 = max(64, self.d // 2)
        h2 = max(64, self.d // 2)
        h3 = max(32, self.d // 4)

        self.head = nn.Sequential(
            nn.Linear(self.d, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, self.n_actions),
        )

        self.apply(self._weight_init)
        self.to(device)

    @staticmethod
    def _weight_init(m):
        if isinstance(m, (nn.Linear, nn.modules.conv.Conv2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.preprocessor(x)
        z = self.perception(x)
        q = self.head(z)
        return q

    @torch.no_grad()
    def act(self, obs, eps: float = 0.05):
        # Ensure batch dimension
        single = False
        if not torch.is_tensor(obs):
            # Preprocessor can handle numpy; just keep flag for output shape
            pass
        else:
            single = (obs.dim() == 3) or (obs.dim() == 1)

        q = self.forward(obs if not single else obs.unsqueeze(0))
        if eps > 0 and torch.rand(1, device=q.device).item() < eps:
            a = torch.randint(self.n_actions, (q.size(0),), device=q.device)
        else:
            a = q.argmax(dim=-1)

        return a[0] if single else a

class RunningMeanStd:
    def __init__(self, epsilon = 1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.count = epsilon
        self.var = np.zeros(shape, 'float64')
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_count = x.shape[0]
        batch_var = np.var(x, axis=0)
        self.update_from_moments(batch_mean, batch_count, batch_var)

    def update_from_moments(self, batch_mean, batch_count, batch_var):
        self.mean, self.count, self.var = self.update_mean_var_count_from_moments(batch_mean, batch_count, batch_var)

    def update_mean_var_count_from_moments(self, batch_mean, batch_count, batch_var):
        delta = batch_mean - self.mean
        var_delta = batch_var - self.var
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        new_var = self.var + var_delta * batch_count / tot_count
        return new_mean, tot_count, new_var
    
def feudal_loss(storage, next_v_m, next_v_w, args, step):
    warmup = 1e6
    # Discount rewards, both of size B x T
    ret_m = next_v_m
    ret_w = next_v_w

    storage.placeholder()  # Fill ret_m, ret_w with empty vals
    for i in reversed(range(args.num_steps)):
        # calculate R = sum(r_i + sum(gamma * R_i-1))
        ret_m = storage.m_r[i] + args.gamma_m * ret_m * storage.m[i]
        ret_w = storage.r[i] + args.gamma_w * ret_w * storage.m[i]
        storage.ret_m[i] = ret_m
        storage.ret_w[i] = ret_w

    # Optionally, normalize the returns
    storage.normalize(['ret_w', 'ret_m'])

    rewards_intrinsic, value_m, value_w, ret_w, ret_m, logps, entropy, \
        state_goal_cosines, goal_entropy = storage.stack(
            ['r_i', 'v_m', 'v_w', 'ret_w', 'ret_m',
             'logp', 'entropy', 's_goal_cos', 'goal_entropy'])

    # Calculate advantages, size B x T
    scale = scale = .1 + (1.0 - .1) * min(1.0, step/warmup)
    r_i = (rewards_intrinsic - rewards_intrinsic.mean() / (rewards_intrinsic.std() + 1e-8)) * scale
    
    # whant to get closer to 0
    advantage_w = (ret_w + args.alpha * r_i) - value_w
    advantage_m = ret_m - value_m 

    loss_worker = (logps * advantage_w.detach()).mean()
    loss_manager = -(state_goal_cosines * advantage_m.detach()).mean()

    # Update the critics into the right direction
    value_w_loss = 0.5 * advantage_w.pow(2).mean()
    value_m_loss = 0.25 * advantage_m.pow(2).mean()

    # want to get larger, how different the actions are and goals are.
    entropy = entropy.mean()
    goal_entropy = goal_entropy.mean()

    loss = - loss_worker - loss_manager + value_w_loss + value_m_loss \
        - (args.entropy_coef * entropy) - ((args.entropy_coef / 5) * goal_entropy)

    return loss, {'loss/total_fun_loss': loss.item(),
                  'loss/worker': loss_worker.item(),
                  'loss/manager': loss_manager.item(),
                  'loss/value_worker': value_w_loss.item(),
                  'loss/value_manager': value_m_loss.item(),
                  'worker/entropy': entropy.item(),
                  'worker/advantage': advantage_w.mean().item(),
                  'worker/intrinsic_reward': rewards_intrinsic.mean().item(),
                  'manager/cosines': state_goal_cosines.mean().item(),
                  'manager/advantage': advantage_m.mean().item(),
                  'manager/entropy': goal_entropy.item()}