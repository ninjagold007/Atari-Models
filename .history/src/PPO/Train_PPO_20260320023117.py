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


class RolloutBuffer:#PPO uses rollout buffer to store transitions before update
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self, device):
        states = torch.cat(self.states).to(device)
        actions = torch.cat(self.actions).to(device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        log_probs = torch.cat(self.log_probs).to(device)
        values = torch.cat(self.values).to(device)
        return states, actions, rewards, dones, log_probs, values


class PPOTrainer:
    def __init__(self, num_envs):
        # make vector envs
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

        # model
        input_shape = (hp.NUM_STACK, hp.FRAME_H, hp.FRAME_W)
        self.policy_net = PPO(input_shape, self.n_actions).to(device)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=hp.LR)

        os.makedirs(hp.SAVE_DIR, exist_ok=True)

        self._reset_all_envs_initial()

    def _reset_all_envs_initial(self):
        frames, _ = self.envs.reset()
        for i in range(self.num_envs):
            pf = preprocess_frame(frames[i]).to(device).float()
            self.stacked_frames[i] = deque([pf]*hp.NUM_STACK, maxlen=hp.NUM_STACK)
            self.current_states[i] = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)

    # select actions from policy
    def select_actions(self):
        actions, log_probs, values = [], [], []

        for i in range(self.num_envs):
            state = self.current_states[i]

            if state is None:
                actions.append(torch.tensor(0, device=device))
                log_probs.append(torch.tensor(0.0, device=device))
                values.append(torch.tensor(0.0, device=device))
            else:
                action, log_prob, value = self.policy_net.get_action_and_value(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

        return torch.stack(actions), torch.stack(log_probs), torch.stack(values)

    # GAE + returns
    def compute_returns_and_advantages(self, rewards, dones, values, next_value):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_adv = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            td_error = rewards[t] + hp.GAMMA * next_val * (1 - dones[t]) - values[t]

            running_adv = td_error + hp.GAMMA * hp.GAE_LAMBDA * running_adv * (1 - dones[t])

            advantages[t] = running_adv
            returns[t] = advantages[t] + values[t]

        return returns, advantages

    #optimize
    def optimize_model(self):
        # merge buffers (VERY important)
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        for b in self.buffers:
            s, a, r, d, lp, v = b.get(device)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            dones.append(d)
            log_probs.append(lp)
            values.append(v)

        states = torch.cat(states)
        actions = torch.cat(actions)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)
        log_probs = torch.cat(log_probs)
        values = torch.cat(values)

        with torch.no_grad():
            next_value = values[-1] * (1 - dones[-1])

        returns, advantages = self.compute_returns_and_advantages(
            rewards, dones, values, next_value
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.size(0)

        for _ in range(hp.PPO_EPOCHS):
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, hp.MINIBATCH_SIZE):
                mb_idx = indices[start:start+hp.MINIBATCH_SIZE]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = log_probs[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, new_values, entropy = self.policy_net.evaluate_actions(
                    mb_states, mb_actions
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(
                    ratio,
                    1.0 - hp.PPO_CLIP_EPSILON,
                    1.0 + hp.PPO_CLIP_EPSILON
                ) * mb_adv

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, mb_returns)
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + hp.VALUE_LOSS_COEF * value_loss
                    - hp.ENTROPY_COEF * entropy_loss
                )

                self.losses = np.append(self.losses, loss.item())
                self.td_errors = np.append(
                    self.td_errors,
                    (mb_returns - new_values.detach()).abs().mean().item()
                )

                self.optimizer.zero_grad()
                loss.backward()

                # gradient clipping (important)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)

                self.optimizer.step()

    def run(self):
        print(f"Starting training with {self.num_envs} envs.")
        finished = 0

        env_start = np.zeros(self.num_envs)
        env_start[:] = time.perf_counter()

        while finished < hp.TOTAL_FINISHED_EPISODES_TO_RUN:
            rollout_steps = 0

            while rollout_steps < hp.ROLLOUT_STEPS:
                actions, log_probs, values = self.select_actions()

                next_frames, rewards, terminated, truncated, _ = self.envs.step(actions.cpu().numpy())
                dones = np.logical_or(terminated, truncated)

                done_indices = []

                for i in range(self.num_envs):
                    r = float(rewards[i])
                    d = bool(dones[i])

                    self.episode_rewards[i] += r

                    pf = preprocess_frame(next_frames[i]).to(device).float()
                    self.stacked_frames[i].append(pf)
                    next_state = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)

                    if self.current_states[i] is not None:
                        self.buffers[i].add(
                            self.current_states[i],
                            actions[i],
                            r,
                            d,
                            log_probs[i].detach(),
                            values[i].detach()
                        )

                    self.current_states[i] = None if d else next_state

                    if actions[i].item() == 2:
                        self.left_count += 1
                    elif actions[i].item() == 3:
                        self.right_count += 1

                    if d:
                        done_indices.append(i)

                self.global_steps += 1
                rollout_steps += 1

                if done_indices:
                    for i in done_indices:
                        frame, _ = self.envs.envs[i].reset()

                        pf = preprocess_frame(frame).to(device).float()
                        self.stacked_frames[i] = deque([pf]*hp.NUM_STACK, maxlen=hp.NUM_STACK)
                        self.current_states[i] = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)

                        if self.episode_rewards[i] > self.best_reward:
                            self.best_reward = self.episode_rewards[i]
                            torch.save(self.policy_net.state_dict(),
                                os.path.join(hp.SAVE_DIR, "policy_net_best.pth"))

                        self.all_scores = np.append(self.all_scores, self.episode_rewards[i])

                        self.episode_rewards[i] = 0
                        self.episode_counters[i] += 1
                        finished += 1

                        print(f"[env {i}] episode {self.episode_counters[i]} total {finished}")

                        if finished % hp.SAVE_EVERY_FINISHED == 0:
                            torch.save(self.policy_net.state_dict(),
                                os.path.join(hp.SAVE_DIR, f"checkpoint_{finished}.pth"))

                        self.episode_lengths = np.append(
                            self.episode_lengths,
                            time.perf_counter() - env_start[i]
                        )
                        env_start[i] = time.perf_counter()

            self.optimize_model()

            for b in self.buffers:
                b.clear()

        print("Training done.")

        moving_avg = np.convolve(np.array(self.all_scores), np.ones((100,))/100, mode='valid')

        make_chart(self.all_scores, "Rewards", "Episode", "Reward")
        make_chart(moving_avg, "Moving Avg", "Episode", "Reward")
        make_chart(self.losses, "Loss", "Step", "Loss")
        make_chart(self.td_errors, "TD Error", "Step", "Error")