import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
from collections import deque
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv
from DQN.DQNModel import DQN
from DQN.ReplayBuffer import ReplayBuffer, Transition
from Preproccessing.Preproccessing import preprocess_frame
from config.hyperparams import params
hp = params()


#CUDA!!!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


class A2CTrainer:
    def __init__(self, num_envs):
         # make vectorized envs
        def make_env(): 
            return gym.make(hp.ENV_ID, render_mode=None)
        self.envs = SyncVectorEnv([make_env for _ in range(num_envs)])
        self.num_envs = num_envs

        # set Action space
        sample_env = self.envs.envs[0]
        self.n_actions = sample_env.action_space.n

        # shape of the game input
        input_shape = (hp.NUM_STACK, hp.FRAME_H, hp.FRAME_W)

    

# Actor (πθ)
#    ↓ chooses
# Action a_t
#    ↓ causes
# Reward r_t, State s_{t+1}
#    ↓ used by
# Critic (Vφ)
#    ↓ computes
# Advantage A_t
#    ↓ updates
# Actor (πθ)
