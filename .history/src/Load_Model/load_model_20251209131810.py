import gymnasium as gym
import ale_py
import torch
from DQN.DQNModel import DQN

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gym.register_envs(ale_py)



class load:
    def __init__(self):
        pass

    def run(self):
        env = gym.make("ALE/Breakout-v5", render_mode="human")
        height, width, channels = env.observation_space.shape
        actions = env.action_space.n

        model = DQN((channels, height, width), actions).to(device)
        model.load_state_dict(torch.load('SavedWeights/policy_net_best_DQN.pth', map_location=device))
        model.eval()  # Set to evaluation mode
loader = load()
loader.run()
