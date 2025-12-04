
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


class DQNTrainer:
    def __init__(self, num_envs=hp.NUM_ENVS):
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
        self.policy_net = DQN(input_shape, self.n_actions).to(device)
        self.target_net = DQN(input_shape, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=hp.LR)

        # Initialize replay buffer
        self.replay = ReplayBuffer(
            hp.REPLAY_CAPACITY,
            num_stack=hp.NUM_STACK,
            frame_h=hp.FRAME_H,
            frame_w=hp.FRAME_W,
            device=device 
        )

        # deque is basically a double-sided queue
        # stacked frames is like a sticky note for each frame
        # must complete an end of the pile before moving to the next note/frame
        # each frame note contains info about memory
        # ✅ Initialize list for stacked frames
        self.stacked_frames = []
        for i in range(num_envs):
            self.stacked_frames.append(deque(maxlen=hp.NUM_STACK))

        # Current states for each environment
        self.current_states = []
        for i in range(num_envs):
            self.current_states.append(None)

        # Other variables
        self.global_steps = 0
        self.episode_counters = [0] * num_envs
        self.episode_rewards = [0.0] * num_envs
        self.best_reward = -float('inf')
        self.left_count = 0
        self.right_count = 0

        # Create save directory if it doesn't exist
        os.makedirs(hp.SAVE_DIR, exist_ok=True)

        # Initialize all environments
        self._reset_all_envs_initial()

    # Reset all envs at the start
    def _reset_all_envs_initial(self):
        frames, _ = self.envs.reset()
        # Initialize stacked frames and current states
        for i in range(self.num_envs):
            # CUDA!!!
            pf = preprocess_frame(frames[i]).to(device).float()
            # Create a deque filled with the initial frame
            self.stacked_frames[i] = deque([pf] * hp.NUM_STACK, maxlen=hp.NUM_STACK)
            # Concatenate stacked frames to form the current state
            self.current_states[i] = torch.cat(list(self.stacked_frames[i]), dim=0).unsqueeze(0)


    # Epsilon-greedy action selection
    #Equation from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    def _epsilon(self):
        return hp.EPS_END + (hp.EPS_START - hp.EPS_END) * math.exp(-1. * self.global_steps / hp.EPS_DECAY)

    # Select actions for all environments
    def select_actions(self):
        # cat = concatenate, dim=0 means along the first dimension (batch dimension)
        state_batch = torch.cat(self.current_states, dim=0)
        
        # with no_grad: picks the action with the highest Q-value without tracking gradients
        with torch.no_grad():
            q_values = self.policy_net(state_batch)
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()

        # epsilon greedy exploration
        # eps is the probability of choosing a random action
        eps = self._epsilon()
        rand_mask = np.random.rand(self.num_envs) < eps

        # Sample random actions for all environments
        random_actions = np.empty(self.num_envs, dtype=int)
        for i in range(self.num_envs):
            random_actions[i] = self.envs.envs[i].action_space.sample()

        # Replace greedy actions with random actions where rand_mask is True
        for i in range(self.num_envs):
            if rand_mask[i]:
                greedy_actions[i] = random_actions[i]

        return greedy_actions

    #optimize by sampling from replay buffer
    def optimize_model(self):
        #check if enough samples are available
        if len(self.replay) < max(hp.BATCH_SIZE, hp.LEARN_START):
            return

        
        transitions = self.replay.sample(hp.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Prepare batches for training
        state_batch = torch.cat(batch.state, dim=0)
        action_batch = torch.cat(batch.action, dim=0)
        reward_batch = torch.cat(batch.reward, dim=0)
        done_batch = torch.cat(batch.done, dim=0)

        # ~ inverts bits, so non_final_mask is True for non-final states
        non_final_mask = ~done_batch.squeeze(1) 
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, done_batch) if not d.item()], dim=0) \
            if non_final_mask.any() else None

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(hp.BATCH_SIZE, device=device)
        if non_final_next_states is not None:
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute the expected Q values
        expected_values = reward_batch.squeeze(1) + hp.GAMMA * next_state_values
        loss = F.smooth_l1_loss(state_action_values.squeeze(1), expected_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Soft update of the target network's weights
        #tp = target parameter, pp = policy parameter
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(tp.data * (1.0 - hp.TARGET_TAU) + pp.data * hp.TARGET_TAU)

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


