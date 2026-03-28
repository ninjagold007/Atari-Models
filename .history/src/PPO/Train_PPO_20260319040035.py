import sys, os

import PPO
from graph_chart_making.makeCharts import make_chart
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import math
from collections import deque
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv
from Preproccessing.Preproccessing import preprocess_frame
from config.hyperparams import params
hp = params()


#CUDA!!!
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

class RolloutBuffer:#PPO uses a rollout buffer to store transitions for multiple steps before updating the policy. This class manages that buffer.
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = [] 
        self.value = []   


    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.value.clear()

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.value.append(value)

    def get_batches(self, device):
        states = torch.cat(self.states).to(device)
        actions = torch.cat(self.actions).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        log_probs = torch.cat(self.log_probs).to(device)
        values = torch.cat(self.value).to(device)
        return states, actions, rewards, dones, log_probs, values


class PPOTrainer:
    def __init__(self, num_envs):
         # make vectorized envs
        def make_env(): 
            return gym.make(hp.ENV_ID, render_mode=None)
        self.envs = SyncVectorEnv([make_env for _ in range(num_envs)])
        self.num_envs = num_envs
        self.buffers = [RolloutBuffer() for _ in range(self.num_envs)]
        self.current_states = [None] * self.num_envs
        self.stacked_frames = [deque(maxlen=hp.NUM_STACK) for _ in range(self.num_envs)]
        self.losses = []
        self.mean_q_values = []
        self.td_errors = []
        self.episode_lengths = []
        self.eps_tracker = []
        self.all_scores = []
        self.global_steps = 0
        self.episode_rewards = np.zeros(self.num_envs)
        self.episode_counters = np.zeros(self.num_envs)
        self.best_reward = -float("inf")
        self.left_count = 0
        self.right_count = 0

        

        # set Action space
        sample_env = self.envs.envs[0]
        self.n_actions = sample_env.action_space.n

        # shape of the game input
        input_shape = (hp.NUM_STACK, hp.FRAME_H, hp.FRAME_W)
        # Initialize critic and actor!
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
        for i in range(self.num_envs):
        pf = preprocess_frame(frames[i]).to(device).float()
        self.stacked_frames[i] = deque([pf]*hp.NUM_STACK, maxlen=hp.NUM_STACK)
        self.current_states[i] = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)
      

    # Epsilon-greedy action selection
    #Equation from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    # def _epsilon(self):
        return hp.EPS_END + (hp.EPS_START - hp.EPS_END) * math.exp(-1. * self.global_steps / hp.EPS_DECAY)

    # Select actions for all environments
    def select_actions(self):
        actions = []
        log_probs = []
        values = []
        for i in range(self.num_envs):
            state = self.current_states[i]
            if state is None:
                # If the environment is done, select a dummy action (e.g., 0) and log_prob/value of 0
                actions.append(torch.tensor(0, device=device))
                log_probs.append(torch.tensor(0.0, device=device))
                values.append(torch.tensor(0.0, device=device))
            else:
                action, log_prob, value = self.policy_net.get_action_and_value(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
        return torch.stack(actions), torch.stack(log_probs), torch.stack(values)

    #optimize
    def optimize_model(self):
        for buffer in self.buffers:
            # Get batches from buffer and compute returns and advantages
            states, actions, rewards, dones, log_probs, values = buffer.get_batches(device)
            returns = self.compute_returns(rewards, dones, values)
            advantages = returns - values

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            for _ in range(hp.PPO_EPOCHS):
                new_log_probs, new_values = self.policy_net.evaluate_actions(states, actions)
                ratio = torch.exp(new_log_probs - log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - hp.PPO_CLIP_EPSILON, 1.0 + hp.PPO_CLIP_EPSILON) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, returns)
                loss = policy_loss + hp.VALUE_LOSS_COEF * value_loss
                self.losses = np.append(self.losses, loss.item())
                self.td_errors= np.append(self.td_errors, (returns - values).abs().mean().item())


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    def compute_returns(self, rewards, dones, values, next_value=0):
        returns = torch.zeros_like(rewards)
        running_return = next_value
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + hp.GAMMA * running_return * (1 - dones[t])
            returns[t] = running_return
        return returns
    def compute_advantages(self, rewards, dones, values, next_value=0):
        advantages = torch.zeros_like(rewards)
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + hp.GAMMA * next_value * (1 - dones[t]) - values[t]
            running_advantage = td_error + hp.GAMMA * hp.VALUE_LOSS_COEF * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage
            next_value = values[t]
        return advantages
    def compute_returns_and_advantages(self, rewards, dones, values, next_value=0):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        running_return = next_value
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            td_error = rewards[t] + hp.GAMMA * next_value * (1 - dones[t]) - values[t]
            running_advantage = td_error + hp.GAMMA * hp.VALUE_LOSS_COEF * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage
            running_return = rewards[t] + hp.GAMMA * running_return * (1 - dones[t])
            returns[t] = running_return
            next_value = values[t]
        return returns, advantages
    def run(self): # log_probs from actor, values from critic, rewards and dones from envs, states from stacked frames
        print(f"Starting training with {self.num_envs} environments (float32 GPU).")
        finished_episode_total = 0

        env_start = np.zeros(self.num_envs)
        env_start[:] = time.perf_counter()

        # Main training loop
        while finished_episode_total < hp.TOTAL_FINISHED_EPISODES_TO_RUN:
            rollout_steps = 0
            while rollout_steps < hp.ROLLOUT_STEPS:#make hp.ROLLOUT_STEPS a multiple of hp.NUM_ENVs for simplicity
                actions, log_probs, values = self.select_actions()
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

                    #rollout the next state if not done, otherwise set to None
                    buffer = self.buffers[i]
                    buffer.add(self.current_states[i], actions[i], reward, done_flag, log_probs[i], values[i])

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
                rollout_steps += 1

                # Handle completed episodes
                if done_indices:
                    frames, _ = self.envs.reset()
                    pf = preprocess_frame(frame).to(device).float()
                    self.stacked_frames[i] = deque([pf]*hp.NUM_STACK, maxlen=hp.NUM_STACK)
                    self.current_states[i] = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)
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
                        self.all_scores = np.append(self.all_scores, self.episode_rewards[i])

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
                        env_start[i] = time.perf_counter()
                        self.episode_lengths = np.append(self.episode_lengths, time.perf_counter() - env_start[i])

            #optimizes after each rollout instead of OPTIMIZE EVERY
            self.optimize_model()
            # Clear buffers after optimization
            for buffer in self.buffers:
                buffer.clear()  
        #done :)
        print("Training completed.")
        moving_avg_reward = np.convolve(np.array(self.all_scores), np.ones((100,))/100, mode='valid')
    
        make_chart( np.array(self.all_scores), "Episode Rewards", "Episode", "Reward")
        make_chart( moving_avg_reward, "Moving Average Reward", "Episode", "Moving Average Reward (100 episodes)")
        make_chart( np.array(self.losses), "Training Loss", "Training Steps", "Loss")
        make_chart( np.array(self.episode_lengths), "Episode Lengths", "Finished Episodes", "Length (time)")
        #make_chart( np.array(self.mean_q_values), "Mean Q-Values", "Training Steps", "Mean Q-Value")
        make_chart( np.array(self.td_errors), "TD Errors", "Training Steps", "Mean TD Error")



