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




class PPOTrainer:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        # Initialize other PPO-specific parameters here