import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformReward
import ale_py
gym.register_envs(ale_py)
from collections import deque, Counter

import torch
import numpy as np

import logging
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import json
# import cv2

class Logger:
    def __init__(self, run_name, args):
        dt = datetime.now()

        #Raw Date time doesn't work as a file name on windows
        ts = dt.strftime("%Y%m%d-%H%M%S")

        if not os.path.exists('logs'):
            os.makedirs('logs')
            os.makedirs('models')

        self.log_dir = os.path.join("logs", f"{run_name}_{ts}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_name = f"{run_name}_{ts}.json"
        self.start_time = time.time()
        self.n_eps = 0
        self.log = dict()

        self.writer = SummaryWriter(self.log_dir)

        # logging.basicConfig(
        #     level=logging.DEBUG,
        #     format='%(asctime)s %(message)s',
        #     handlers=[
        #         logging.StreamHandler(),
        #         logging.FileHandler(f'{self.log_name}.log'),
        #         ],
        #     datefmt='%Y/%m/%d %I:%M:%S %p')
        # logging.info(args)

    def log_scalars(self, scalar_dict, step):
        for key, val in scalar_dict.items():
            self.writer.add_scalar(key, val, step)

    def log_episode(self, info, step):
        if info is not None:
            self.n_eps += 1
            time_expired = (time.time()-self.start_time) / 60 / 60
            self.log[self.n_eps] = {'total_reward' : info['total_reward'].tolist(), 'episode_frame_number': info['episode_frame_number'].tolist(), 'time_expired': time_expired}
            # logging.info(f"> ep = {self.n_eps} | total steps = {step}"
            #                  f" | reward = {reward} | length = {length}"
            #                  f" | hours = {time_expired:.3f}")
    
    def save(self):
        with open(f'logs/{self.log_name}', 'w') as r:
            json.dump(self.log, r)


class Storage:
    def __init__(self, size, keys = None):
        if keys is None:
            keys = []
        self.keys = keys
        self.size = size
        self.reset()
    
    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self,k).append(v)
    
    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)
    
    def reset(self):
        for key in self.keys:
            setattr(self, key, [])
    
    def normalize(self, keys):
        for key in keys:
            k = torch.stack(getattr(self, key))
            k = (k - k.mean()) / (k.std() + 1e-10)
            setattr(self, key, [i for i in k])
    
    def stack(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.stack(x, dim=0), data)

class PacmanRewardWrapper(gym.Wrapper):
    def __init__(self, env, rnd_model, alpha = .01, stagnation_penalty=.01, death_penalty=20, pellet_bonus=.1, stagnation_penalty_enable = True):
        super().__init__(env)

        self.pellet_bonus = pellet_bonus
        self.death_penalty = death_penalty
        self.stagnation_penalty_enable = stagnation_penalty_enable
        self.ram = self._check_ram_observation()
        self.lives = 0

        self.prev_score = 0
        self.total_reward = 0
        self.prev_positon = None
        self.score_best = 0
        self.episode = -1
        self.position_history = deque(maxlen=300)

        self.rnd_model = rnd_model
        self.alpha = alpha
        self.device = rnd_model.device
    
    def _check_ram_observation(self):
        if  self.env.spec.kwargs['obs_type'] == 'ram':
            return True
        else:
            return False
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.prev_score = 0
        self.episode +=1
        self.prev_positon = self._get_position(obs)
        self.lives = info['lives']
        self.total_reward = 0
        self.delay_counter = 0
        self.position_history = deque(maxlen=300)
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_score = self.total_reward + reward
        self.total_reward += reward
        current_position = self._get_position(obs)
        stagnate = True

        modified_reward = reward / 100
        if reward == 10:
            # reward pellet collection
            modified_reward += self.pellet_bonus

        if current_position not in self.position_history:
            # Reward new positions
            self.position_history.append(current_position)
            modified_reward += .1
            stagnate = False

        stagnation_penalty = 0

        if self.stagnation_penalty_enable and current_position and stagnate and current_position != (88,98):
            # penalty for not moving around
            position_count = self.position_history.count(current_position)
            if position_count < 25:
                stagnation_penalty = 0
            else:
                position_len = len(self.position_history)
                position_count /= position_len
                stagnation_penalty = position_count * .1
    
            modified_reward -= stagnation_penalty
            self.position_history.append(current_position)

        if info['lives'] < self.lives :
            # penalty for death
            self.delay_counter = 0
            death_penalty =  self.death_penalty
            modified_reward -= death_penalty
            self.lives = info['lives']
            if self.lives == 0:
                modified_reward -= self.death_penalty * 2/5

        modified_reward += min(.1, 0.001 + 1e-4 * self.total_reward) 
        # alive reward
        
        if self.score_best < self.total_reward:
            # reward for new high score
            modified_reward += .1
            self.score_best = self.total_reward

        if self.episode > 5:
            # reward for new states
            last_N = [current_position == self.position_history[i] if len(self.position_history) > 10 else False for i in range(-10,0)]
            if sum(last_N) == 0:
                beta = max(1, 1000 / self.episode) * .01
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                r_i = self.rnd_model.intrinsic_reward(obs_tensor).detach().cpu().numpy()

                modified_reward += self.alpha * r_i.item() *  beta


        self.prev_score = current_score
        self.prev_positon = current_position


        info.update({
            'total_reward': self.total_reward,
            'original_reward': reward,
            'modified_reward': modified_reward,
            'position': current_position,
            'score': current_score,
            'using_ram': self.ram,
            'episode': self.episode,
            'score best': self.score_best
        })

        return obs, modified_reward, terminated, truncated, info
    
    def _get_position(self, obs):
        if self.ram and len(obs) >= 128:
            try:
                x = obs[10]
                y = obs[16]
                return (x,y)
            except IndexError:
                return None
        else:
            return self._detect_pacman_position_pixel(obs)
    
    def _get_score(self, obs, info):
        if isinstance(info, dict):
            if 'score' in info:
                return info['score']
            elif hasattr(info, 'episode') and 'r' in info['episode']:
                return info['episode']['r']
        
        if self.ram and len(obs) >= 128:
            try:
              return obs[120] * 256 + obs[121]
            except IndexError:
                return 0  
            
    def _detect_pacman_position_pixel(self, obs):
        """Detect Ms. Pacman position from pixels"""
        if len(obs.shape) == 3:  # RGB
            # Look for yellow Ms. Pacman sprite
            yellow_mask = (obs[:, :, 0] > 200) & (obs[:, :, 1] > 200) & (obs[:, :, 2] < 100)
            if np.any(yellow_mask):
                y_coords, x_coords = np.where(yellow_mask)
                # Filter for reasonably sized sprites
                if len(x_coords) > 5 and len(x_coords) < 100:
                    return (int(np.mean(x_coords)), int(np.mean(y_coords)))
        elif len(obs.shape) == 2:  # Grayscale
            # Look for bright sprite
            bright_mask = obs > 200
            if np.any(bright_mask):
                y_coords, x_coords = np.where(bright_mask)
                if len(x_coords) > 5 and len(x_coords) < 100:
                    return (int(np.mean(x_coords)), int(np.mean(y_coords)))
        
        return None

class MontezumaRewardWrapper(gym.Wrapper):
    def __init__(self, env, rnd_model, exploration_bonus=0.1, stagnation_penalty=0.01, death_penalty=50, alpha =.1):
        super().__init__(env)
        self.exploration_bonus = exploration_bonus
        self.stagnation_penalty = stagnation_penalty
        self.death_penalty = death_penalty
        
        self.visited_rooms = set()
        self.prev_position = None
        self.prev_score = 0
        self.total_reward = 0
        self.lives = 0
        self.position_history = deque(maxlen=500)
        self.rnd_model = rnd_model
        self.device = rnd_model.device

        self.alpha = alpha

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.visited_rooms.clear()
        self.prev_position = self._get_position(obs)
        self.prev_score = self._get_score(obs)
        self.lives = self._get_lives(obs)
        self.total_reward = 0
        self.position_history.clear()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        score = self._get_score(obs)
        position = self._get_position(obs)
        lives = self._get_lives(obs)

        modified_reward = 0.0

        # Extrinsic: Score increase
        if score > self.prev_score:
            modified_reward += (score - self.prev_score) / 100.0
        self.prev_score = score

        # Exploration: new room
        room = self._get_room(obs)
        if room not in self.visited_rooms:
            self.visited_rooms.add(room)
            modified_reward += self.exploration_bonus

        # Exploration: new position (very lightweight)
        if position not in self.position_history:
            self.position_history.append(position)
            modified_reward += 0.01

        # Stagnation penalty
        else:
            modified_reward -= self.stagnation_penalty

        # Death penalty
        if lives < self.lives:
            modified_reward -= self.death_penalty
        self.lives = lives

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        r_i = self.rnd_model.intrinsic_reward(obs_tensor).detatch().cpu().numpy()[0]
        r_i = (r_i - np.mean(r_i)) / (np.std(r_i) + 1e-8)

        modified_reward += self.alpha * r_i

        self.total_reward += reward
        info.update({
            "modified_reward": modified_reward,
            "score": score,
            "room": room,
            "position": position,
            "lives": lives,

        })

        return obs, modified_reward, terminated, truncated, info

    # --- RAM helpers ---

    def _get_score(self, obs):
        # Montezuma’s score is usually stored in bytes 56 & 57 (depends on ALE version)
        try:
            return obs[56] * 256 + obs[57]
        except IndexError:
            return 0

    def _get_room(self, obs):
        # Room number typically at byte 3
        return int(obs[3])

    def _get_position(self, obs):
        # Player X/Y positions typically bytes 42 (x) and 43 (y)
        try:
            return (int(obs[42]), int(obs[43]))
        except IndexError:
            return None

    def _get_lives(self, obs):
        # Lives often at byte 58
        try:
            return int(obs[58])
        except IndexError:
            return 0


class VectorEnvVisualizer:
    def __init__(self, env_idx=0, save_videos=True, save_dir='videos'):
        self.env_idx = env_idx  # Which environment to visualize (0 to num_envs-1)
        self.save_videos = save_videos
        self.save_dir = save_dir
        self.frames = []
        self.episode_count = 0
        self.best_reward = 0
        self.best_rewards_x = []
        self.best_reward_y = []
        
        if save_videos:
            os.makedirs(save_dir, exist_ok=True)
        
        # Setup real-time display
        plt.ion()
        self.fig, ((self.ax1, self.ax2, self.ax3), (self.ax4, self.ax5, self.ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(15, 6))
        self.im1 = None
        self.im2 = None
        
    def capture_frame(self, envs, step, actions, rewards, terminated, truncated,  info):
        """Capture frame from vector environment"""
        try:
            # Get frames from all environments
            frames = envs.call('render')
            
            if frames is None or len(frames) <= self.env_idx:
                return
                
            # Focus on one environment
            frame = frames[self.env_idx]
            if frame is None:
                return
                
            self.frames.append(frame)
            
            # Display the selected environment
            if self.im1 is None:
                self.im1 = self.ax1.imshow(frame)
                self.ax1.set_title(f'Environment {self.env_idx}')
                self.ax1.axis('off')
            else:
                self.im1.set_array(frame)
            
            # Show info for this specific environment
            action = actions[self.env_idx] if hasattr(actions, '__len__') else actions
            reward = rewards[self.env_idx] if hasattr(rewards, '__len__') else rewards
            done = terminated[self.env_idx] or truncated[self.env_idx]
            
            self.ax1.set_title(f'Env {self.env_idx} | Step: {step} | Action: {action} | Reward: {reward:.2f} | Done: {done}')
            
            # Show action distribution visualization (if you want)
            self.visualize_action_dist(actions, rewards, info, step)
            
            plt.pause(0.01)
            
            # Save episode when done
            # if done and self.save_videos:
                # self.save_episode_video()
                
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def capture_frame_test(self, envs, step, actions, rewards, terminated, truncated,  info):
        """Capture frame from vector environment"""
        try:
            # Get frames from all environments
            frames = envs.call('render')
            
            if frames is None or len(frames) <= self.env_idx:
                return
                
            # Focus on one environment
            frame = frames[self.env_idx]
            if frame is None:
                return
                
            self.frames.append(frame)
            
            # Display the selected environment
            if self.im1 is None:
                self.im1 = self.ax1.imshow(frame)
                self.ax1.set_title(f'Environment {self.env_idx}')
                self.ax1.axis('off')
            else:
                self.im1.set_array(frame)
            
            # Show info for this specific environment
            action = actions[self.env_idx] if hasattr(actions, '__len__') else actions
            reward = rewards[self.env_idx] if hasattr(rewards, '__len__') else rewards
            done = terminated[self.env_idx] or truncated[self.env_idx]
            
            self.ax1.set_title(f'Env {self.env_idx} | Step: {step} | Action: {action} | Reward: {reward:.2f} | Done: {done}')
            
            # Show action distribution visualization (if you want)
            # self.visualize_action_dist(actions, rewards, info, step)
            
            plt.pause(0.01)
            
            # Save episode when done
            # if done and self.save_videos:
                # self.save_episode_video()
                
        except Exception as e:
            print(f"Visualization error: {e}")
    
    
    def visualize_action_dist(self, actions, rewards, info, step):
        """Show action distribution across all environments"""
        if hasattr(actions, '__len__'):
            # Bar chart of actions taken across all environments
            unique_actions, counts = np.unique(actions, return_counts=True)
            quarter = len(info['total_reward']) // 4
            IQM = np.median(np.sort(info['total_reward'])[quarter:-quarter])
            if np.sum(info['lives'] == 0) > 0 or IQM > self.best_reward:
                self.best_reward_y.append(IQM)
                self.best_rewards_x.append(step)
                if IQM > self.best_reward:
                    self.best_reward = IQM

            self.episode_count += 1
            
            self.ax2.clear()
            self.ax2.bar(unique_actions, counts, alpha=0.7)
            self.ax2.set_title('Action Distribution Across Environments')
            self.ax2.set_xlabel('Action')
            self.ax2.set_ylabel('Count')
            
            self.ax3.clear()
            self.ax3.bar(list(range(len(rewards))),info['total_reward'], alpha=0.7)
            self.ax3.set_title('Rewards Across Environments')
            self.ax3.set_xlabel('total reward')
            self.ax3.set_ylabel('envorment')

            self.ax4.clear()
            self.ax4.scatter(self.best_rewards_x, self.best_reward_y)
            self.ax4.set_title('Best IQM across time')
            self.ax4.set_xlabel('step')
            self.ax4.set_ylabel('best IQM')


            self.ax5.clear()
            self.ax5.bar(np.arange(len(info['total_reward'])), info['score best'])
            # self.ax5.set_xlabel('episode')
            self.ax5.set_ylabel('best score')

            self.ax6.clear()
            self.ax6.bar(np.arange(len(info['total_reward'])), info['episode'])
            # self.ax5.set_xlabel('episode')
            self.ax6.set_ylabel('episode')

            # Add reward info
            mean_reward = np.mean(rewards) if hasattr(rewards, '__len__') else rewards
            self.ax2.text(0.7, 0.9, f'Mean Reward: {mean_reward:.3f}', 
                         transform=self.ax2.transAxes, fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # def save_episode_video(self, fps=10):
    #     """Save episode as video"""
    #     if not self.frames or not self.save_videos:
    #         return
            
    #     height, width = self.frames[0].shape[:2]
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    #     video_path = f'{self.save_dir}/env_{self.env_idx}_episode_{self.episode_count}.mp4'
    #     out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
    #     for frame in self.frames:
    #         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #         out.write(frame_bgr)
        
    #     out.release()
    #     print(f"Saved episode video: {video_path}")
        
    #     self.frames = []
    #     self.episode_count += 1

def make_env(env_name, rnd_model, obs, wrapper):
    def _thunk():
        env = gym.make(env_name, render_mode='rgb_array', obs_type=obs)
        env = wrapper(env, rnd_model)
        return env
    return _thunk


def make_envs(env_name, num_envs, args, train=True, rnd_model=None):
    if args.mlp == 1:
        obs = 'ram'
    else:
        obs = "rgb"
    if 'Pacman' in args.env_name:
        envs = gym.vector.SyncVectorEnv([make_env(env_name, rnd_model, obs, PacmanRewardWrapper) for _ in range(num_envs)])
    else:
        envs = gym.vector.SyncVectorEnv(make_env(env_name, rnd_model, obs, MontezumaRewardWrapper))
    envs.reset(seed=args.seed)
    return envs

def take_action(a, eps):
    dist = torch.distributions.Categorical(a)
    if np.random.rand() < eps:
        action = dist.sample()
    else:
        action = torch.argmax(a, dim=-1)
    logp = dist.log_prob(action)
    entropy = dist.entropy()
    return action.cpu().detach().numpy(), logp, entropy
        
