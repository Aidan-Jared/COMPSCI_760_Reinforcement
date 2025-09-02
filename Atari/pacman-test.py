import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import datetime
import numpy as np
from collections import deque
import random
import ale_py
# import renderlab as rl

gym.register_envs(ale_py)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class CNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7)
        self.fc1 = nn.Linear(in_features=1792, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=256)
        self.fc4 = nn.Linear(in_features=256, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=9)
    
    def forward(self, x):
        x = x.to(device)
        x = F.relu(F.max_pool2d(self.convolution1(x),3))
        x = F.relu(F.max_pool2d(self.convolution2(x),3))
        x = F.relu(F.max_pool2d(self.convolution3(x),3,2))
        x = x.reshape(x.size(0), - 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class DQNAgent:
    def __init__(self, action_size = 9, reward_number = 0.37):
        self.state_size = 4
        self.action_size = action_size
        self.memory_n = deque(maxlen=2000)
        self.memory_p = deque(maxlen=2000)
        self.gamma = 1. # discount rate
        self.epsilon = 1. #exploration rate
        self.epsilon_min = .1
        self.epsilon_decay = .9999
        self.learning_rate = .1
        self.model = CNN().to(device)
        self.reward_number = reward_number
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        if reward == 0:
            self.memory_p.append((state, action, reward, next_state, done))
        else:
            self.memory_n.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.from_numpy(state).float()
        act_values = self.model(state_tensor).cpu().detach().numpy()
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(self.memory_n) > batch_size / 2:
            minibatch_n = random.sample(self.memory_n,5)
            minibatch_p = random.sample(self.memory_p,59)
            minibatch = random.sample((minibatch_p + minibatch_n), batch_size)
        else:
            minibatch = random.sample(self.memory_p, batch_size)
        for state, action, reward, next_state, done in minibatch:
            ns_model = self.model(torch.from_numpy(next_state).float()).cpu().detach().numpy()
            if reward == 0:
                reward = 1.0001
                target = reward * np.amax(ns_model[0])
                target_f = ns_model
                target_f[0][action] = target
            else:
                reward = self.reward_number
                target = reward * np.amin(ns_model[0])
                target_max = .0001 * np.amax(ns_model[0])
                target_f = ns_model
                target_f[0][action] = target
                target_f[0][random.choice([i for i in range(0,9) if i not in [action]])] = target_max
            self.train(next_state, target_f, epochs = 1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, input, target, epochs = 1):
        input = torch.from_numpy(input).float().to(device)
        target = torch.from_numpy(target).float().to(device)
        for t in range(epochs):
            y_pred = self.model(input).to(device)
            loss = - self.criterion(y_pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
    def load_all(self, name):
        loaded = torch.load(name)
        self.memory_n = loaded['memory_n']
        self.memory_p = loaded['memory_p']
        self.model.load_state_dict(loaded['state'])
        
    def save_all(self, name):
        torch.save({'state': self.model.state_dict(),
                    'memory_n': self.memory_n,
                    'memory_p': self.memory_p
                   }, name)
        
    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        
    def save(self, name):
        torch.save(self.model.state_dict(), name)

if __name__ == "__main__":
    env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent()
    terminated = False
    truncated = False
    batch_size = 32

    Episodes = 20
    for e in range(Episodes):
        state, info = env.reset()
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        img_plot = ax.imshow(env.render())
        ax.set_title('Ms. Pacman')
        ax.axis('off')
        state = np.reshape(state, (1, 210,160,3)).transpose(0,3,1,2)/255
        for time in range(1000000000):
            if time % 4 == 0:
                frame = env.render()
                img_plot.set_array(frame)
                plt.pause(0.01)
            action = agent.act(state)
            next_state, reward, terminated, truncated , _ = env.step(action)
            reward = reward if not terminated or truncated else 10
            reward = reward if reward == 0 else 10
            next_state = np.reshape(state, (1, 210,160,3)).transpose(0,3,1,2)/255
            agent.remember(state, action, reward, next_state, terminated)
            state = next_state
            if terminated or truncated:
                print("episode: {}/{}, score: {}, e: {:.2}".format(e+1, Episodes , time, agent.epsilon))
                break
            if (len(agent.memory_p) > batch_size) & (len(agent.memory_n) > batch_size/2) :
                agent.replay(batch_size)