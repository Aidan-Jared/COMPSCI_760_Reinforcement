import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from utils import VectorEnvVisualizer, Storage, take_action
from feudalnet import feudal_loss
from collections import deque



class Train:
    def __init__(self, args, model, optimizer, envs, logger, rnd):
        self.model = model
        self.optimizer = optimizer
        self.max_steps = args.max_steps
        self.num_steps = args.num_steps
        self.num_workers = args.num_workers
        self.device = args.device
        self.envs = envs
        self.logger = logger
        self.args = args
        self.rnd = rnd
        self.rnd_optimizer = torch.optim.Adam(self.rnd.parameters(), 1e-3)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 1, .1, 1e4)

    def train_feudal(self):
        eps = self.args.eps
        save_steps =  list(torch.arange(0, int(self.max_steps), int(self.max_steps) // 10).numpy())
        goals, states, masks = self.model.init_obj()
        x, info = self.envs.reset(seed=self.args.seed)
        step = 0
        visualizer = VectorEnvVisualizer(env_idx=0, save_videos=False)
        while step < self.max_steps:
            self.model.repackage_hidden()
            goals = [g.detach() for g in goals]
            storage = Storage(size=self.args.num_steps,
                            keys=['r', 'm_r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                    's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                    'adv_m', 'adv_w', 'goal_entropy', 'obs'])

            for _ in range(self.num_steps):
                action_dist, goals, states, value_m, value_w = self.model(x, goals, states, masks[-1])
                action, logp, entropy = take_action(action_dist, eps)
                x, reward, terminated, truncated, info = self.envs.step(action)
                if step % 160 == 0:
                    visualizer.capture_frame(self.envs, step, action, reward, terminated, truncated, info)
                self.logger.log_episode(info, step)

                mask = torch.FloatTensor(1 - (terminated + truncated)).unsqueeze(-1).to(self.device)
                masks.pop(0)                    
                masks.append(mask)

                storage.add({
                    'r': torch.FloatTensor(reward).unsqueeze(-1).to(self.device),
                    'm_r': torch.FloatTensor(info['original_reward']).unsqueeze(-1).to(self.device) / 100,
                    'r_i': self.model.intrinsic_reward(states, goals, masks),
                    'v_w': value_w,
                    'v_m': value_m,
                    'logp': logp.unsqueeze(-1),
                    'entropy': entropy.unsqueeze(-1),
                    's_goal_cos': self.model.state_goal_cosine(states, goals, masks),
                    'goal_entropy' :self.model.goal_entropy(goals, masks),
                    'm': mask,
                    'obs': torch.Tensor(x).to(self.device)
                })


                if step % 640 == 0 and eps > self.args.decay_limit:
                    # reduce random exploration
                    eps *= self.args.decay
                    self.model.eps_decay()

                step += self.num_workers
            with torch.no_grad():
                # predict the reward of the next step
                *_, next_v_m, next_v_w = self.model(x, goals, states, mask, save = False)
                next_v_m = next_v_m.detach()
                next_v_w = next_v_w.detach()

        
            self.optimizer.zero_grad()
            loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, self.args, step)
            loss.backward()
            # update model with gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()

            self.lr_scheduler.step()


            obs_batch = torch.stack(storage.obs)

            pred, target = self.rnd(obs_batch)

            rnd_loss = F.mse_loss(pred, target)

            self.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            self.rnd_optimizer.step()

            self.logger.log_scalars(loss_dict, step)
            if len(save_steps) > 0 and step > save_steps[0]:
                    torch.save({
                        'model': self.model.state_dict(),
                        'args': self.args,
                        'processor_mean': self.model.preprocessor.rms.mean,
                        'optim': self.optimizer.state_dict()},
                        f'models/{self.args.env_name[4:]}_{self.args.run_name}_step={step}.pt')
                    save_steps.pop(0)
        self.envs.close()
        torch.save({
        'model': self.model.state_dict(),
        'args': self.args,
        'processor_mean': self.model.preprocessor.rms.mean,
        'optim': self.optimizer.state_dict()},
        f'models/{self.args.env_name}_{self.args.run_name}_steps={step}.pt')
        
    def train_transformer(self):
        eps = self.args.eps
        save_steps =  list(torch.arange(0, int(self.max_steps), int(self.max_steps) // 10).numpy())
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.max_steps / self.num_steps)
        goals, states, masks, actions, rewards, zs, frames = self.model.init_obj()
        cos_mask = masks.copy()
        x, info = self.envs.reset(seed=self.args.seed)
        step = 0
        visualizer = VectorEnvVisualizer(env_idx=0, save_videos=False)
        scalar = torch.amp.GradScaler(self.device)
        batch_idx = 0
        while step < self.max_steps:
            self.model.repackage_hidden()
            goals, states, zs, actions, rewards, frames = self.model.detach_sequences(goals, states, zs, actions, rewards, frames)
            storage = Storage(size=self.num_steps,
                            keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                    's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                    'adv_m', 'adv_w', 'kl_loss'])

            for _ in range(self.num_steps):
                action_dist, goals, states, zs, value_m, value_w = self.model(x, zs, actions, rewards, goals, states, frames, masks)
                action, logp, entropy = take_action(action_dist, eps)
                x, reward, terminated, truncated, info = self.envs.step(action)
                actions.pop(0)
                rewards.pop(0)
                frames.pop(0)
                actions.append(torch.FloatTensor(action).unsqueeze(1).to(self.device))
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                frames.append(torch.FloatTensor(info['episode_frame_number']).unsqueeze(1).to(self.device))
                if step % 160 == 0:
                    visualizer.capture_frame(self.envs, step, action, reward, terminated, truncated, info)
                self.logger.log_episode(info, step)
                mask = torch.FloatTensor(1 - (terminated + truncated)).unsqueeze(-1).to(self.device)
                masks.pop(0)
                cos_mask.pop(0)
                with torch.no_grad():
                    if torch.sum(mask) < self.num_workers:
                        for idx, i in enumerate(masks):
                            masks[idx] = i * mask
                        for idx, i in enumerate(cos_mask[:-self.args.time_horizon]):
                            cos_mask[idx + self.args.time_horizon] = i * mask
                    
                masks.append(mask)
                cos_mask.append(mask)

                storage.add({
                    'r': torch.FloatTensor(reward).unsqueeze(-1).to(self.device),
                    'r_i': self.model.intrinsic_reward(states, goals, cos_mask),
                    'v_w': value_w,
                    'v_m': value_m,
                    'logp': logp.unsqueeze(-1),
                    'entropy': entropy.unsqueeze(-1),
                    's_goal_cos': self.model.state_goal_cosine(states, goals, cos_mask),
                    'm': mask,
                })

                step += self.args.num_workers

                if step % 640 == 0 and eps > self.args.decay_limit:
                    # reduce random exploration
                    eps *= self.args.decay
                    self.model.eps_decay()
                
                if step % 5000 == 0:
                    # shuffel goals to reduce enviroment specific overfitting
                    if len(goals) > 1:
                        perm = torch.randperm(self.num_workers)
                        goals[-1] = goals[-1][perm]

            with torch.no_grad():
                *_, next_v_m, next_v_w = self.model(x, zs, actions, rewards, goals, states, frames, masks, save = False)
                next_v_m = next_v_m.detach()
                next_v_w = next_v_w.detach()
            
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type):
                loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, self.args)
            scalar.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            if batch_idx % 20 == 0 and self.device == 'cuda':
                torch.cuda.synchronize()
                time.sleep(2)
            batch_idx += 1
            # scheduler.step()
            self.logger.log_scalars(loss_dict, step)
            if len(save_steps) > 0 and step > save_steps[0]:
                torch.save({
                    'model': self.model.state_dict(),
                    'args': self.args,
                    'processor_mean': self.model.preprocessor.rms.mean,
                    'optim': self.optimizer.state_dict()},
                    f'models/{self.args.env_name[4:]}_{self.args.run_name}_step={step}.pt')
                self.logger.save()
                save_steps.pop(0)

        self.envs.close()
        torch.save({
        'model': self.model.state_dict(),
        'args': self.args,
        'processor_mean': self.model.preprocessor.rms.mean,
        'optim': self.optimizer.state_dict()},
        f'models/{self.args.env_name}_{self.args.run_name}_steps={step}.pt')

    def train_qmodel(self):
        q_model_trainer = QModelTrainer(
            args=self.args,
            model=self.model,
            optimizer=self.optimizer,
            envs=self.envs,
            logger=self.logger,
            rnd=self.rnd
        )
        q_model_trainer.train_qmodel()

# A simple replay buffer class
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        transitions = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = np.stack(states)
        actions = np.array(actions)
        rewards = np.array(rewards).astype(np.float32)
        next_states = np.stack(next_states)
        dones = np.array(dones)

        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)
    
class QModelTrainer(Train):
    def __init__(self, **kwargs):
        super().__init__(**kwargs) #args, model, optimizer, envs, logger, device
        self.device = self.args.device
        self.max_steps = getattr(self.args, "max_steps", 1_000_000)
        self.num_workers = self.envs.num_envs
        self.num_steps = getattr(self.args, "num_steps", 256)
        self.batch_size = getattr(self.args, "batch_size", 32)
        self.replay_buffer_capacity = getattr(self.args, "replay_buffer_capacity", 500_000)
        self.replay_buffer_min_size = getattr(self.args, "replay_buffer_min_size", 10_000)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_capacity)


    @torch.no_grad()
    def _epsilon_greedy(self, qvals, eps, n_actions):
        """
        qvals: Tensor [B, A] on any device
        returns: np.ndarray[int] of shape [B]
        """
        b = qvals.size(0)
        greedy = qvals.argmax(dim=1).detach().cpu().numpy()
        if eps <= 0.0:
            return greedy
        mask = np.random.rand(b) < eps
        random_act = np.random.randint(0, n_actions, size=b)
        return np.where(mask, random_act, greedy)

    def train_qmodel(self):
        args = self.args
        device = self.device
        
        gamma = getattr(args, "gamma", getattr(args, "gamma_w", 0.99))
        eps = getattr(args, "eps", 1.0)
        eps_decay = getattr(args, "decay", 0.9999)
        eps_limit = getattr(args, "decay_limit", 0.05)
        target_update = getattr(args, "target_update", 10_000) 
        grad_clip = getattr(args, "grad_clip", 0.5)

        self.model.train()
        target_net = copy.deepcopy(self.model).to(device).eval()
        for p in target_net.parameters():
            p.requires_grad_(False)
        
        # Reset envs
        x, info = self.envs.reset(seed=getattr(args, "seed", None))
        step = 0
        updates = 0
        
        visualizer = VectorEnvVisualizer(env_idx=0, save_videos=False)

        # Initial fill of replay buffer with random actions
        print("Filling replay buffer...")
        while len(self.replay_buffer) < self.replay_buffer_min_size:
            actions = np.random.randint(0, self.model.n_actions, size=self.num_workers)
            x_next, reward, terminated, truncated, info = self.envs.step(actions)
            for i in range(self.num_workers):
                mask = 1.0 - (terminated[i] + truncated[i])
                self.replay_buffer.add(x[i], actions[i], reward[i], x_next[i], terminated[i] or truncated[i])
            x = x_next
            step += self.num_workers
            if step % 5000 == 0:
                print(f"Buffer size: {len(self.replay_buffer)}/{self.replay_buffer_min_size}")

        print("Replay buffer filled. Starting training...")
        
        # Main training loop
        while step < self.max_steps:
            # Collect data with epsilon-greedy policy
            qvals = self.model(x)                             # x: np.ndarray from vector env
            save_steps = list(torch.arange(0, int(self.max_steps), int(self.max_steps) // 10).numpy())
            actions = self._epsilon_greedy(qvals, eps, self.model.n_actions)

            x_next, reward, terminated, truncated, infos = self.envs.step(actions)

            # ---- Correctly build transitions, using final_observation for dones
            dones = np.logical_or(terminated, truncated)
            final_obs = infos.get("final_observation", None)

            for i in range(self.num_workers):
                next_si = final_obs[i] if (dones[i] and final_obs is not None) else x_next[i]
                self.replay_buffer.add(x[i], actions[i], reward[i], next_si, bool(dones[i]))

            # Immediately reset finished envs so the rollout continues cleanly
            if np.any(dones):
                reset_indices = np.where(dones)[0]

                # Try the per-index reset; some envs ignore 'indices' and return full batch
                reset_obs, reset_infos = self.envs.reset(seed=None, options={"indices": reset_indices})

                # Normalize tuple return (obs, info) patterns if your env uses them
                if isinstance(reset_obs, tuple):
                    reset_obs = reset_obs[0]

                # If the env returned a full batch, replace whole x_next; otherwise, just the subset.
                if reset_obs.shape == x_next.shape:
                    # full-batch reset
                    x_next = reset_obs
                else:
                    # subset reset (shape should be [len(indices), ...])
                    x_next[reset_indices] = reset_obs

            x = x_next
            x_next, reward, terminated, truncated, info = self.envs.step(actions)

            # Make sure the logger populates info['total_reward'], etc. *before* you draw
            self.logger.log_episode(info, step)
            if step % 160 == 0:
                visualizer.capture_frame(self.envs, step, actions, reward, terminated, truncated, info)

            step += self.num_workers

            # Sample a batch from the replay buffer
            batch = self.replay_buffer.sample(self.batch_size)
            if batch is None:
                continue

            states_np, actions_np, rewards_np, next_states_np, dones_np = batch
            
            # 1) Build tensors with correct dtype/device
            #    If observations are pixels (uint8), cast to float32 and scale to [0,1].
            obs_is_uint8 = states_np.dtype == np.uint8

            states_t = torch.as_tensor(states_np, device=device, dtype=torch.float32)
            next_states_t = torch.as_tensor(next_states_np, device=device, dtype=torch.float32)
            if obs_is_uint8:
                states_t      = states_t / 255.0
                next_states_t = next_states_t / 255.0

            actions_t = torch.as_tensor(actions_np.reshape(-1, 1), device=device, dtype=torch.long)
            rewards_t = torch.as_tensor(rewards_np.reshape(-1, 1), device=device, dtype=torch.float32)
            masks_t   = torch.as_tensor(1.0 - dones_np.reshape(-1, 1), device=device, dtype=torch.float32)

            # (Optional but common in Atari) Reward clipping for stability
            rewards_t = rewards_t.clamp_(-1.0, 1.0)

            # 2) Targets
            with torch.no_grad():
                # ---- Standard DQN target:
                q_next = target_net(next_states_t)                       # [B, A]
                q_next_max = q_next.max(dim=1, keepdim=True).values      # [B, 1]
                target = rewards_t + gamma * masks_t * q_next_max        # [B, 1]

                # ---- Double DQN target (recommended):
                # next_actions = self.model(next_states_t).argmax(dim=1, keepdim=True)  # online picks
                # q_next_tgt = target_net(next_states_t).gather(1, next_actions)        # target evaluates
                # target = rewards_t + gamma * masks_t * q_next_tgt

            # 3) Prediction: Q(s, a)
            q_pred_all = self.model(states_t)                     # [B, A]
            q_pred     = q_pred_all.gather(1, actions_t)          # [B, 1]
            
            loss = F.smooth_l1_loss(q_pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            updates += 1

            # Periodic hard target sync
            if updates % target_update == 0:
                target_net.load_state_dict(self.model.state_dict())
            
            # Epsilon decay
            if eps > eps_limit:
                eps_by_frame = lambda t: max(eps_limit, 1.0 - (1.0 - eps_limit) * (t / 1_000_000))  # linear to 1M steps
                eps = eps_by_frame(step)

            # Logging
            with torch.no_grad():
                avg_q = q_pred.mean().item()
                avg_r = float(np.mean(rewards_np))
                self.logger.log_scalars({
                    "q_loss": loss.item(),
                    "q_avg": avg_q,
                    "reward/mean": avg_r,
                    "train/epsilon": eps,
                    "train/updates": updates,
                }, step)
                
            # Checkpoint (simplified for brevity)
            
            if step % 50000 == 0:
                print(f"Checkpoint at step {step}")
                torch.save({
                    'model': self.model.state_dict(),
                    'args': self.args,
                    'processor_mean': getattr(self.model, "preprocessor", None).rms.mean if hasattr(self.model, "preprocessor") else None,
                    'optim': self.optimizer.state_dict()
                }, f'models/{self.args.env_name[4:] if len(self.args.env_name) > 4 else self.args.env_name}_{self.args.run_name}_step={step}.pt')
                self.logger.save()
                save_steps.pop(0)

        self.envs.close()
        # Final save
        if not os.path.exists('models/ALE/'):
            os.makedirs('models/ALE/')
        torch.save({
            'model': self.model.state_dict(),
            'args': self.args,
            'processor_mean': getattr(self.model, "preprocessor", None).rms.mean if hasattr(self.model, "preprocessor") else None,
            'optim': self.optimizer.state_dict()
        }, f'models/{self.args.env_name}_{self.args.run_name}_steps={step}.pt')