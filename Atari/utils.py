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
        self.log_name = dt.replace(second=0, microsecond=0)
        self.start_time = time.time()
        self.n_eps = 0
        self.log = dict()

        if not os.path.exists('logs'):
            os.makedirs('logs')
            os.makedirs('models')
        self.writer = SummaryWriter(self.log_name)

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

class StagnationPenalty:
    def __init__(self, stagnation_pen = .0001, position_history_size = 1000, stagnation_threshold = .8, penalty_escalation = 1.2, max_mult = 1000, delay=65):
        self.stagnation_pen = stagnation_pen
        self.position_history_size = position_history_size
        self.stagnation_threshold = stagnation_threshold
        self.penalty_escalation = penalty_escalation
        self.max_multi = max_mult
        self.delay = delay
        self.delay_counter = 0
        
        self.position_history = deque(maxlen=self.position_history_size)
        self.counter = 0
        self.current_multi = 1.0

    def calculate_stagnation_penalty(self, current_position):
        if current_position is None or self.delay_counter < self.delay:
            self.delay_counter += 1
            return 0.0
        
        self.position_history.append(current_position)

        if len(self.position_history) < self.position_history_size // 2:
            return 0.0
        
        unique_positions = len(set(self.position_history))
        diversity_ratio = unique_positions / len(self.position_history)


        penalty = 0

        if diversity_ratio < self.stagnation_threshold:
            self.current_multi *= self.penalty_escalation

            penalty = self.stagnation_pen * min(self.max_multi, self.current_multi)

        else:
            self.counter = max(0, self.counter-5)
            self.current_multi = max(1, self. current_multi / self.penalty_escalation)
        self.delay_counter += 1

        return penalty
    
    def reset_stagnation_tracking(self):
        self.position_history.clear()
        self.counter = 0
        self.current_multi = 1

class Reward:
    def _init__(self, gamma = .99):
        self.prev_reward = 0
        self.gamma = gamma
    
    def maximum_reward(self, score):
        if score > 0:
            reward = max(score, self.gamma * self.prev_reward)
        else:
            reward = min(score, self.gamma * self.prev_reward)
        self.prev_reward = reward
        return reward
    
    def diff_reward(self, score):
        reward = abs(score - self.gamma * self.prev_reward)
        self.prev_reward = reward
        return reward


class PacmanRewardWrapper(gym.Wrapper, StagnationPenalty, Reward):
    def __init__(self, env, stagnation_penalty=.01, pellet_bonus=1, death_penalty=1000, stagnation_penalty_enable = True):
        super().__init__(env)
        Reward._init__(self,gamma=.99)
        if stagnation_penalty_enable:
            StagnationPenalty.__init__(self,stagnation_pen=stagnation_penalty, position_history_size=750, stagnation_threshold=.75, penalty_escalation=1.2)

        self.pellet_bonus = pellet_bonus
        self.death_penalty = death_penalty
        self.stagnation_penalty_enable = stagnation_penalty_enable
        self.ram = self._check_ram_observation()
        self.lives = 0
        self.pellet_history = deque(maxlen=1000)

        self.prev_score = 0
        self.total_reward = 0
        self.prev_positon = None
        self.corridor_history = deque(maxlen=300)
        self.courner_time_counter = 0
        self.alive = 0
    
    def _check_ram_observation(self):
        if  self.env.spec.kwargs['obs_type'] == 'ram':
            return True
        else:
            return False
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.pellet_history = deque(maxlen=1000)
        self.prev_score = 0
        self.prev_positon = self._get_position(obs)
        self.corridor_history.clear()
        self.lives = info['lives']
        self.courner_time_counter = 0
        self.total_reward = 0
        self.delay_counter = 0
        if self.stagnation_penalty_enable:
            self.reset_stagnation_tracking()
        
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_score = self.total_reward + reward
        self.total_reward += reward
        current_position = self._get_position(obs)
        # current_score = self._get_score(obs, info)

        modified_reward = np.clip(reward,0,200)
        self.pellet_history.append(current_score)
        if current_score > self.prev_score:
            unique = len(set(self.pellet_history))
            pellet_bonus = unique / len(self.pellet_history) + self.pellet_bonus
            score_increase = current_score - self.prev_score
            modified_reward += (pellet_bonus * score_increase) * info['lives'] #+ self.alive
            self.prev_score = current_score
            self.prev_positon = current_position

            info.update({
                'total_reward': self.total_reward,
                'original_reward': reward,
                'modified_reward': modified_reward,
                'position': current_position,
                'score': current_score,
                'using_ram': self.ram
            })
            return obs, modified_reward, terminated, truncated, info


        stagnation_penalty = 0
        corner_penalty = 0
        corridor_penalty = 0

        if self.stagnation_penalty_enable and current_position:
            stagnation_penalty = self.calculate_stagnation_penalty(current_position)
            modified_reward -= stagnation_penalty

            if self._is_corner_positon(current_position,  obs):
                self.courner_time_counter += .1
                if self.courner_time_counter > 20:
                    corner_penalty = .2 * (self.courner_time_counter)
                    modified_reward -= corner_penalty
            else:
                self.courner_time_counter = max(0,  self.courner_time_counter - 2)
            
            self.corridor_history.append(current_position)
            if len(self.corridor_history) >= 20:
                if self._detect_corridor_oscillation():
                    corridor_penalty = .01
                    modified_reward -= corridor_penalty 
                    # if self.alive > 0:
                    #     self.alive -= corridor_penalty
            if info['lives'] < self.lives :
                # self.alive = 0
                self.delay_counter = 0
                death_penalty = min(100, max(5, info['episode_frame_number'] / 1000))
                modified_reward -= death_penalty
                self.lives = info['lives']
                self.pellet_history = deque(maxlen=1000)

        # modified_reward += self.alive
        self.prev_score = current_score
        self.prev_positon = current_position

        # modified_reward = self.maximum_reward(modified_reward)

        info.update({
            'total_reward': self.total_reward,
            'original_reward': reward,
            'modified_reward': modified_reward,
            'position': current_position,
            'score': current_score,
            'using_ram': self.ram
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
    
    def _is_corner_positon(self, position, obs):
        if not position:
            return False
        
        x, y = position

        if self.ram:
            return (x < 20 or x > 140 or y < 20 or y > 180)
        else:
            # For pixel version, check screen boundaries
            obs_height, obs_width = obs.shape[:2] if len(obs.shape) >= 2 else (210, 160)
            return (x < obs_width * 0.1 or x > obs_width * 0.9 or 
                   y < obs_height * 0.1 or y > obs_height * 0.9)
        
    def _detect_corridor_oscillation(self):
        if len(self.corridor_history) < 10:
            return False

        positions = list(self.corridor_history)[-10:]
        positions = [p for p in positions if p is not None]

        if len(positions) < 6:
            return False
        
        x_coords = [float(pos[0]) for pos in positions]
        y_coords = [float(pos[1]) for pos in positions]
        
        x_changes = sum(1 for i in range(1, len(x_coords)) if abs(x_coords[i] - x_coords[i-1]) > 5)
        y_changes = sum(1 for i in range(1, len(y_coords)) if abs(y_coords[i] - y_coords[i-1]) > 5)

        return (x_changes > 5 or y_changes > 5)
    
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


class ReturnWrapper(gym.Wrapper):
    #######################################################################
    # Copyright (C) 2020 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
    # Permission given to modify the code as long as you keep this        #
    # declaration at the top                                              #
    #######################################################################
    def __init__(self, env):
        super().__init__(env)
        self.total_rewards = 0
        self.steps = 0
        self.death_penalty = 50
        self.current_lives = None
        self.time_reward = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.current_lives is not None and self.current_lives > info['lives']:
            reward -= self.death_penalty * info['episode_frame_number'] / 10000
        else:
            reward += self.time_reward
            if info['episode_frame_number'] < 2000 and reward < 11:
                reward *= info['episode_frame_number'] / 100
            elif info['episode_frame_number'] > 2000 and reward < 11:
                reward *= 2
        self.total_rewards += reward
        self.steps += 1
        self.current_lives = info['lives']
        info['total_rewards'] = self.total_rewards
        if terminated or truncated:
            info['returns/episodic_reward'] = self.total_rewards
            info['returns/episodic_length'] = self.steps
            self.total_rewards = 0
            self.steps = 0
        else:
            info['returns/episodic_reward'] = 0
            info['returns/episodic_length'] = 0
        return obs, reward, terminated, truncated, info

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
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(1, 4, figsize=(15, 6))
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

def atari_wraper(env):
    # env = AtariPreprocessing(env, grayscale_obs=False, scale_obs=True, frame_skip=1)
    # env = ReturnWrapper(env)
    env = PacmanRewardWrapper(env)
    return env

def make_envs(env_name, num_envs, args, train=True):
    # 
    if args.mlp == 1:
        envs = gym.make_vec(env_name, num_envs, wrappers=[atari_wraper], vectorization_mode="sync", render_mode='rgb_array', obs_type='ram')
    else:
        envs = gym.make_vec(env_name, num_envs, wrappers=[atari_wraper], vectorization_mode="sync", render_mode='rgb_array')
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
        
