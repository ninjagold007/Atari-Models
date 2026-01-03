import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import ale_py
import torch
from DQN.DQNModel import DQN
from config.hyperparams import params
hp = params()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gym.register_envs(ale_py)



class load:
    def __init__(self):
        pass

    def run(self):
        env = gym.make("ALE/Breakout-v5", render_mode="human")
        height, width, channels = env.observation_space.shape
        actions = env.action_space.n

        model = DQN((hp.NUM_STACK, hp.FRAME_H, hp.FRAME_W), actions).to(device)
        model.load_state_dict(torch.load('C:\\Users\\ninja\\School\\Atari-Models\\src\\SavedWeights\\policy_net_best_DQN.pth', map_location=device))
        model.eval()  # Set to evaluation mode

        obs, _ = env.reset()                # 🔹 1
        done = False

        while not done:                     # 🔹 2
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                action = model(obs_t).argmax(dim=1).item()

            obs, _, terminated, truncated, _ = env.step(action)  # 🔹 3
            done = terminated or truncated
loader = load()
loader.run()
