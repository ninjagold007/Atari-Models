import sys, os
import time
import math
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Preproccessing.Preproccessing import preprocess_frame
from config.hyperparams import params
from graph_chart_making.makeCharts import make_chart
import matplotlib.pyplot as plt
from PPO.PPOModel import PPO

hp = params()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


class RolloutBuffer:  # stores rollouts before PPO updates
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
        self.buffers = [RolloutBuffer() for _ in range(num_envs)]
        self.current_states = [None] * num_envs
        self.stacked_frames = [deque(maxlen=hp.NUM_STACK) for _ in range(num_envs)]

        self.losses = []
        self.td_errors = []
        self.episode_lengths = []
        self.all_scores = []

        self.global_steps = 0
        self.episode_rewards = np.zeros(num_envs)
        self.episode_counters = np.zeros(num_envs)
        self.best_reward = -float("inf")
        self.left_count = 0
        self.right_count = 0

        # action space
        sample_env = self.envs.envs[0]
        self.n_actions = sample_env.action_space.n

        # input shape for CNN
        input_shape = (hp.NUM_STACK, hp.FRAME_H, hp.FRAME_W)
        self.policy_net = PPO(input_shape, self.n_actions).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=hp.LR)

        # save dir
        os.makedirs(hp.SAVE_DIR, exist_ok=True)

        self._reset_all_envs_initial()

    def _reset_all_envs_initial(self):
        frames, _ = self.envs.reset()
        for i in range(self.num_envs):
            pf = preprocess_frame(frames[i]).to(device).float()
            self.stacked_frames[i] = deque([pf] * hp.NUM_STACK, maxlen=hp.NUM_STACK)
            self.current_states[i] = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)

    # select actions from policy
    def select_actions(self):
        actions, log_probs, values = [], [], []
        for i in range(self.num_envs):
            state = self.current_states[i]
            if state is None:  # env done
                actions.append(torch.tensor(0, device=device))
                log_probs.append(torch.tensor(0.0, device=device))
                values.append(torch.tensor(0.0, device=device))
            else:
                a, lp, v = self.policy_net.get_action_and_value(state)
                actions.append(a.squeeze().unsqueeze(0))
                log_probs.append(lp.unsqueeze(0))
                values.append(v.unsqueeze(0))
        return torch.stack(actions), torch.stack(log_probs), torch.stack(values)

    # optimize PPO
    def optimize_model(self):
        for buffer in self.buffers:
            states, actions, rewards, dones, log_probs, values = buffer.get_batches(device)

            with torch.no_grad():
                next_value = values[-1]

            returns, advantages = self.compute_returns_and_advantages(rewards, dones, values, next_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(hp.PPO_EPOCHS):
                new_log_probs, new_values, entropy = self.policy_net.evaluate_actions(states, actions)
                ratio = torch.exp(new_log_probs - log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - hp.PPO_CLIP_EPSILON, 1 + hp.PPO_CLIP_EPSILON) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, returns)
                entropy_loss = entropy.mean()

                loss = policy_loss + hp.VALUE_LOSS_COEF * value_loss - hp.ENTROPY_COEF * entropy_loss

                self.losses = np.append(self.losses, loss.item())
                self.td_errors = np.append(self.td_errors, (returns - values).abs().mean().item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    # compute returns & advantages (GAE)
    def compute_returns_and_advantages(self, rewards, dones, values, next_value=0):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            next_val = next_value if t == len(rewards) - 1 else values[t + 1]
            td_error = rewards[t] + hp.GAMMA * next_val * (1 - dones[t]) - values[t]
            running_advantage = td_error + hp.GAMMA * hp.GAE_LAMBDA * running_advantage * (1 - dones[t])
            advantages[t] = running_advantage
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    # main training loop
    def run(self):
        print(f"Starting training with {self.num_envs} envs (float32 GPU).")
        finished_episode_total = 0
        env_start = np.zeros(self.num_envs)
        env_start[:] = time.perf_counter()

        while finished_episode_total < hp.TOTAL_FINISHED_EPISODES_TO_RUN:
            rollout_steps = 0
            while rollout_steps < hp.ROLLOUT_STEPS:
                actions, log_probs, values = self.select_actions()
                actions_np = actions.squeeze(-1).cpu().numpy().astype(np.int32)
                next_frames, rewards, terminateds, truncateds, _ = self.envs.step(actions_np)
                dones = np.logical_or(terminateds, truncateds)
                done_indices = []

                for i in range(self.num_envs):
                    reward = float(rewards[i])
                    done_flag = bool(dones[i])
                    self.episode_rewards[i] += reward

                    pf = preprocess_frame(next_frames[i]).to(device).float()
                    self.stacked_frames[i].append(pf)
                    next_state = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)

                    buffer = self.buffers[i]
                    if self.current_states[i] is not None:
                        buffer.add(
                            self.current_states[i],
                            actions[i],
                            reward,
                            done_flag,
                            log_probs[i].detach(),
                            values[i].detach()
                        )

                    self.current_states[i] = None if done_flag else next_state

                    if actions[i].item() == 2:
                        self.left_count += 1
                    elif actions[i].item() == 3:
                        self.right_count += 1

                    if done_flag:
                        done_indices.append(i)

                self.global_steps += 1
                rollout_steps += 1

                # reset envs after done
                if done_indices:
                    for i in done_indices:
                        frame, _ = self.envs.envs[i].reset()
                        pf = preprocess_frame(frame).to(device).float()
                        self.stacked_frames[i] = deque([pf] * hp.NUM_STACK, maxlen=hp.NUM_STACK)
                        self.current_states[i] = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)

                        if self.episode_rewards[i] > self.best_reward:
                            self.best_reward = self.episode_rewards[i]
                            torch.save(self.policy_net.state_dict(),
                                       os.path.join(hp.SAVE_DIR, "policy_net_best.pth"))
                            print(f"[env {i}] New best reward: {self.best_reward:.1f}")

                        self.all_scores = np.append(self.all_scores, self.episode_rewards[i])
                        self.episode_counters[i] += 1
                        self.episode_rewards[i] = 0.0
                        finished_episode_total += 1

                        print(f"[env {i}] Finished episode {self.episode_counters[i]}, total finished: {finished_episode_total}")
                        print(f"Left: {self.left_count}, Right: {self.right_count}")

                        if finished_episode_total % hp.SAVE_EVERY_FINISHED == 0:
                            torch.save(self.policy_net.state_dict(),
                                       os.path.join(hp.SAVE_DIR, f"policy_net_finished_{finished_episode_total}.pth"))
                            torch.save(self.policy_net.state_dict(),
                                       os.path.join(hp.SAVE_DIR, f"policy_net_best_at_{finished_episode_total}.pth"))

                        self.episode_lengths = np.append(self.episode_lengths, time.perf_counter() - env_start[i])
                        env_start[i] = time.perf_counter()

            # optimize after rollout
            self.optimize_model()
            for buffer in self.buffers:
                buffer.clear()

        print("Training completed.")

        moving_avg_reward = np.convolve(np.array(self.all_scores), np.ones((100,)) / 100, mode='valid')
        make_chart(np.array(self.all_scores), "Episode Rewards", "Episode", "Reward")
        make_chart(moving_avg_reward, "Moving Average Reward", "Episode", "Moving Average Reward (100 episodes)")
        make_chart(np.array(self.losses), "Training Loss", "Training Steps", "Loss")
        make_chart(np.array(self.episode_lengths), "Episode Lengths", "Finished Episodes", "Length (time)")
        make_chart(np.array(self.td_errors), "TD Errors", "Training Steps", "Mean TD Error")