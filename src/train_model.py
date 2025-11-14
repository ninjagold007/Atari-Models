import gymnasium as gym
from model1 import DQN
import ale_py
import torch
gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class train:
    def __init__(self):
        pass

    def run(self):
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        observation = env.reset(seed=42)
        height, width, channels = env.observation_space.shape
        actions = env.action_space.n
        score = 0
        done = False

        model = DQN((channels, height, width), actions).to(device)
        # env.close()
        torch.save(model.state_dict(), '/SavedWeights/dqn_weights.pth')

trainer = train()
trainer.run()
