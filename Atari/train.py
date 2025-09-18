import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from utils import VectorEnvVisualizer, Storage, take_action
from feudalnet import feudal_loss

class Train:
    def __init__(self, args, model, optimizer, envs, logger):
        self.model = model
        self.optimizer = optimizer
        self.max_steps = args.max_steps
        self.num_steps = args.num_steps
        self.num_workers = args.num_workers
        self.device = args.device
        self.envs = envs
        self.logger = logger
        self.args = args

    def train_feudal(self):
        eps = self.args.eps
        save_steps =  list(torch.arange(0, int(self.max_steps), int(self.max_steps) // 10).numpy())
        goals, states, masks = self.model.init_obj()
        x, info = self.envs.reset()
        step = 0
        visualizer = VectorEnvVisualizer(env_idx=0, save_videos=False)
        while step < self.max_steps:
            self.model.repackage_hidden()
            goals = [g.detach() for g in goals]
            storage = Storage(size=self.args.num_steps,
                            keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                    's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                    'adv_m', 'adv_w'])

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
                    'r_i': self.model.intrinsic_reward(states, goals, masks),
                    'v_w': value_w,
                    'v_m': value_m,
                    'logp': logp.unsqueeze(-1),
                    'entropy': entropy.unsqueeze(-1),
                    's_goal_cos': self.model.state_goal_cosine(states, goals, masks),
                    'm': mask
                })
                if step % 1280 == 0 and eps > self.args.decay_limit and step > 1e6:
                    eps *= self.args.decay

                step += self.num_workers
            with torch.no_grad():
                *_, next_v_m, next_v_w = self.model(x, goals, states, mask, save = False)
                next_v_m = next_v_m.detach()
                next_v_w = next_v_w.detach()
            self.optimizer.zero_grad()
            loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, self.args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
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
        'optim': self.ptimizer.state_dict()},
        f'models/{self.args.env_name}_{self.args.run_name}_steps={step}.pt')
        
    def train_transformer(self):
        eps = self.args.eps
        save_steps =  list(torch.arange(0, int(self.max_steps), int(self.max_steps) // 10).numpy())
        goals, states, masks, actions, rewards, zs = self.model.init_obj()
        x, info = self.envs.reset(seed=self.args.seed)
        step = 0
        visualizer = VectorEnvVisualizer(env_idx=0, save_videos=False)
        scalar = torch.amp.GradScaler(self.device)
        batch_idx = 0
        while step < self.max_steps:
            self.model.repackage_hidden()
            goals, states, zs, actions, rewards = self.model.detach_sequences(goals, states, zs, actions, rewards)
            storage = Storage(size=self.num_steps,
                            keys=['r', 'r_i', 'v_w', 'v_m', 'logp', 'entropy',
                                    's_goal_cos', 'mask', 'ret_w', 'ret_m',
                                    'adv_m', 'adv_w'])

            for _ in range(self.num_steps):
                action_dist, goals, states, zs, value_m, value_w = self.model(x, zs, actions, rewards, goals, states, masks)
                action, logp, entropy = take_action(action_dist, eps)
                x, reward, terminated, truncated, info = self.envs.step(action)
                actions.pop(0)
                rewards.pop(0)
                actions.append(torch.FloatTensor(action).unsqueeze(1).to(self.device))
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                if step % 160 == 0:
                    visualizer.capture_frame(self.envs, step, action, reward, terminated, truncated, info)
                self.logger.log_episode(info, step)
                mask = torch.FloatTensor(1 - (terminated + truncated)).unsqueeze(-1).to(self.device)
                masks.pop(0)
                # with torch.no_grad():
                #     if torch.sum(mask) < self.num_workers:
                #         for idx, i in enumerate(masks):
                #             masks[idx] = i * mask
                    
                masks.append(mask)

                storage.add({
                    'r': torch.FloatTensor(reward).unsqueeze(-1).to(self.device),
                    'r_i': self.model.intrinsic_reward(states, goals, masks),
                    'v_w': value_w,
                    'v_m': value_m,
                    'logp': logp.unsqueeze(-1),
                    'entropy': entropy.unsqueeze(-1),
                    's_goal_cos': self.model.state_goal_cosine(states, goals, masks),
                    'm': mask
                })

                step += self.args.num_workers

                if step % 1280 == 0 and eps > self.args.decay_limit:
                    eps *= self.args.eps_decay

            with torch.no_grad():
                *_, next_v_m, next_v_w = self.model(x, zs, actions, rewards, goals, states, masks, save = False)
                next_v_m = next_v_m.detach()
                next_v_w = next_v_w.detach()
            
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device.type):
                loss, loss_dict = feudal_loss(storage, next_v_m, next_v_w, self.args)
            scalar.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
            if batch_idx % 20 == 0:
                torch.cuda.synchronize()
                time.sleep(2)
            batch_idx += 1
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
        eps_decay_freq = getattr(args, "eps_decay_freq", 1280)  
        target_update = getattr(args, "target_update", 10_000)  
        grad_clip = getattr(args, "grad_clip", 0.5)

        # Target network
        target_net = copy.deepcopy(self.model).to(device).eval()
        for p in target_net.parameters():
            p.requires_grad_(False)

        # Save schedule
        save_steps = list(torch.arange(0, int(self.max_steps), max(1, int(self.max_steps) // 10)).numpy())

        # Reset envs
        x, info = self.envs.reset(seed=getattr(args, "seed", None))
        step = 0
        updates = 0

        # (Optional) visualizer 
        try:
            visualizer = VectorEnvVisualizer(env_idx=0, save_videos=False)
        except Exception:
            visualizer = None

        while step < self.max_steps:
            # Collect a rollout of length num_steps (on-policy buffer)
            states_buf, acts_buf, rews_buf, masks_buf, next_states_buf = [], [], [], [], []

            for _ in range(self.num_steps):
                # ε-greedy action from current Q
                qvals = self.model(x)  # [B, A]
                actions = self._epsilon_greedy(qvals, eps, self.model.n_actions)

                # Step envs
                x_next, reward, terminated, truncated, info = self.envs.step(actions)

                # Optional viz/log
                if visualizer and step % 160 == 0:
                    visualizer.capture_frame(self.envs, step, actions, reward, terminated, truncated, info)
                self.logger.log_episode(info, step)

                # Done mask
                mask = 1 - (terminated + truncated)  # np array [B]

                # Store transition (raw obs; model will preprocess)
                states_buf.append(x)
                acts_buf.append(actions)
                rews_buf.append(reward)
                masks_buf.append(mask)
                next_states_buf.append(x_next)

                # Eps decay
                step += self.num_workers
                if step % eps_decay_freq == 0 and eps > eps_limit:
                    eps *= eps_decay

                # Advance
                x = x_next

            # ======= Train step over the collected batch =======
            # Flatten time and workers into one big batch
            # Each element is an np array shaped [B, ...]; stack along time then reshape
            def _stack_time(xlist):
                # xlist length = num_steps; each element shape [B, ...]
                # Return np array [T*B, ...]
                return np.concatenate(xlist, axis=0)

            states_np      = _stack_time(states_buf)
            next_states_np = _stack_time(next_states_buf)
            actions_np     = _stack_time([a.reshape(-1, 1) for a in acts_buf])     # [TB, 1]
            rewards_np     = _stack_time([r.reshape(-1, 1) for r in rews_buf]).astype(np.float32)  # [TB,1]
            masks_np       = _stack_time([m.reshape(-1, 1) for m in masks_buf]).astype(np.float32) # [TB,1]

            # Convert actions/rewards/masks to tensors; states are fed as numpy through model (it handles device)
            actions_t = torch.as_tensor(actions_np, device=device, dtype=torch.long)
            rewards_t = torch.as_tensor(rewards_np, device=device)
            masks_t   = torch.as_tensor(masks_np, device=device)

            # Targets: r + γ * mask * max_a' Q_target(s', a')
            with torch.no_grad():
                q_next = target_net(next_states_np)                     # [TB, A]
                q_next_max = q_next.max(dim=1, keepdim=True).values     # [TB, 1]
                target = rewards_t + gamma * masks_t * q_next_max       # [TB, 1]

            # Prediction: Q(s, a)
            q_pred_all = self.model(states_np)                          # [TB, A]
            q_pred = q_pred_all.gather(1, actions_t)                    # [TB, 1]

            loss = F.smooth_l1_loss(q_pred, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            updates += 1

            # Periodic hard target sync
            if (step // self.num_workers) * self.num_workers % target_update == 0:
                target_net.load_state_dict(self.model.state_dict())

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

            # Checkpoint
            if len(save_steps) > 0 and step > save_steps[0]:
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
        torch.save({
            'model': self.model.state_dict(),
            'args': self.args,
            'processor_mean': getattr(self.model, "preprocessor", None).rms.mean if hasattr(self.model, "preprocessor") else None,
            'optim': self.optimizer.state_dict()
        }, f'models/{self.args.env_name}_{self.args.run_name}_steps={step}.pt')
