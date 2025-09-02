import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, TransformReward
from vizdoom import gymnasium_wrapper
import ale_py
gym.register_envs(ale_py)

import torch
import numpy as np

import logging
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
# import cv2

class Logger:
    def __init__(self, run_name, args):
        dt = datetime.now()
        self.log_name = dt.replace(second=0, microsecond=0)
        self.start_time = time.time()
        self.n_eps = 0

        if not os.path.exists('logs'):
            os.makedirs('logs')
            os.makedirs('models')
        self.writer = SummaryWriter(self.log_name)

        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'{self.log_name}.log'),
                ],
            datefmt='%Y/%m/%d %I:%M:%S %p')
        logging.info(args)

    def log_scalars(self, scalar_dict, step):
        for key, val in scalar_dict.items():
            self.writer.add_scalar(key, val, step)

    def log_episode(self, info, step):
        if info is not None:
            self.n_eps += 1
            self.log_scalars(info, step)
            reward = info['returns/episodic_reward']
            length = info['returns/episodic_length']
            time_expired = (time.time()-self.start_time) / 60 / 60
            logging.info(f"> ep = {self.n_eps} | total steps = {step}"
                             f" | reward = {reward} | length = {length}"
                             f" | hours = {time_expired:.3f}")

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

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_rewards += reward
        self.steps += 1
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
        
        if save_videos:
            os.makedirs(save_dir, exist_ok=True)
        
        # Setup real-time display
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        self.im1 = None
        self.im2 = None
        
    def capture_frame(self, envs, step, actions, rewards, terminated, truncated):
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
            self.visualize_action_dist(actions, rewards)
            
            plt.pause(0.01)
            
            # Save episode when done
            # if done and self.save_videos:
                # self.save_episode_video()
                
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def visualize_action_dist(self, actions, rewards):
        """Show action distribution across all environments"""
        if hasattr(actions, '__len__'):
            # Bar chart of actions taken across all environments
            unique_actions, counts = np.unique(actions, return_counts=True)
            
            self.ax2.clear()
            self.ax2.bar(unique_actions, counts, alpha=0.7)
            self.ax2.set_title('Action Distribution Across Environments')
            self.ax2.set_xlabel('Action')
            self.ax2.set_ylabel('Count')
            
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
    env = ReturnWrapper(env)
    env = TransformReward(env, lambda r: np.sign(r))
    return env

def make_envs(env_name, num_envs, seed = 0):
    envs = gym.make_vec(env_name, num_envs, wrappers=[atari_wraper], vectorization_mode="sync", render_mode='rgb_array')
    envs.reset(seed=seed)
    return envs

def take_action(a):
    dist = torch.distributions.Categorical(a)
    action = dist.sample()
    logp = dist.log_prob(action)
    entropy = dist.entropy()
    return action.cpu().detach().numpy(), logp, entropy