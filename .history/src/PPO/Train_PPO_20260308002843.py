import sys, os

import PPO
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
        # Initialize networks
        self.policy_net = PPO(input_shape, self.n_actions).to(device)
   

        # Initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=hp.LR)

        # Create save directory if it doesn't exist
        os.makedirs(hp.SAVE_DIR, exist_ok=True)

        # Initialize all environments
        self._reset_all_envs_initial()

        # Reset all envs at the start
    def _reset_all_envs_initial(self):
        frames, _ = self.envs.reset()
      

    # Epsilon-greedy action selection
    #Equation from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def _epsilon(self):
        return hp.EPS_END + (hp.EPS_START - hp.EPS_END) * math.exp(-1. * self.global_steps / hp.EPS_DECAY)

    # Select actions for all environments
    def select_actions(self):
        return 1

    #optimize by sampling from replay buffer
    def optimize_model(self):
        return 1
       

    def run(self):
        print(f"Starting training with {self.num_envs} environments (float32 GPU).")
        finished_episode_total = 0

        # Main training loop
        while finished_episode_total < hp.TOTAL_FINISHED_EPISODES_TO_RUN:
            actions = self.select_actions()
            next_frames, rewards, terminateds, truncateds, _ = self.envs.step(actions)
            # done is when either terminated or truncated is True
            #logical_or performs element-wise OR operation
            dones = np.logical_or(terminateds, truncateds)

            done_indices = []

            # Process each environment's transition
            for i in range(self.num_envs):
                reward = float(rewards[i])
                done_flag = bool(dones[i])
                self.episode_rewards[i] += reward

                pf = preprocess_frame(next_frames[i]).to(device).float()
                self.stacked_frames[i].append(pf)
                next_state = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)

                # Store transition in replay buffer
                self.replay.push(
                    self.current_states[i],
                    int(actions[i]),
                    None if done_flag else next_state,
                    reward,
                    done_flag
                )

                self.current_states[i] = None if done_flag else next_state

                # Count left/right actions for statistics
                if actions[i] == 2: 
                    self.left_count += 1
                elif actions[i] == 3: 
                    self.right_count += 1

                # Check for episode completion
                if done_flag: 
                    done_indices.append(i)

            # Increment global step and optimize model
            self.global_steps += 1
            if self.global_steps % hp.OPTIMIZE_EVERY == 0:
                self.optimize_model()

            # Handle completed episodes
            if done_indices:
                frames, _ = self.envs.reset()
                for i in done_indices:
                    # Save best model
                    if self.episode_rewards[i] > self.best_reward:
                        self.best_reward = self.episode_rewards[i]
                        torch.save(self.policy_net.state_dict(),
                                   os.path.join(hp.SAVE_DIR, "policy_net_best.pth"))
                        print(f"[env {i}] New best reward: {self.best_reward:.1f}")

                    # Reinitialize stacked frames and current state
                    pf = preprocess_frame(frames[i]).to(device).float()
                    self.stacked_frames[i] = deque([pf]*hp.NUM_STACK, maxlen=hp.NUM_STACK)
                    self.current_states[i] = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)

                    self.episode_counters[i] += 1
                    self.episode_rewards[i] = 0.0
                    finished_episode_total += 1

                    #log stats
                    print(f"[env {i}] Finished episode {self.episode_counters[i]}, total finished: {finished_episode_total}")
                    print(f"Left: {self.left_count}, Right: {self.right_count}")

                    # Save checkpoints at intervals
                    if finished_episode_total % hp.SAVE_EVERY_FINISHED == 0:
                        torch.save(self.policy_net.state_dict(),
                                   os.path.join(hp.SAVE_DIR, f"policy_net_finished_{finished_episode_total}.pth"))
                        print(f"Saved checkpoint at {finished_episode_total} finished episodes.")
                        #also include current best
                        torch.save(self.policy_net.state_dict(),
                                   os.path.join(hp.SAVE_DIR, f"policy_net_best_at_{finished_episode_total}.pth"))
                        

        print("Training completed.")
