import gymnasium as gym
import minari
import ale_py
gym.register_envs(ale_py)
from gymnasium.utils import play

if __name__ == "__main__":
    env = minari.DataCollector(gym.make('MsPacmanNoFrameskip-v4',  render_mode='rgb_array'))
    env.reset()
    play.play(env, zoom=3)
    env._save_to_disk('data')