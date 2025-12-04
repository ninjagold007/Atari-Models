# DQN/ReplayBuffer.py

from collections import namedtuple
import torch
import random

# Transition tuple to store experience steps
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity, num_stack, frame_h, frame_w, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.full = False

        # 4D tensors: [capacity, C, H, W], float32
        self.states = torch.zeros((capacity, num_stack, frame_h, frame_w), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.int64, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, num_stack, frame_h, frame_w), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)

    # Add a transition to the buffer
    def push(self, state, action, next_state, reward, done):
        self.states[self.position] = state.float()
        self.actions[self.position] = torch.tensor([[action]], dtype=torch.int64, device=self.device)
        self.rewards[self.position] = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        self.next_states[self.position] = state.float() if next_state is None else next_state.float()
        self.dones[self.position] = torch.tensor([[done]], dtype=torch.bool, device=self.device)

        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    # Sample a batch of transitions
    def sample(self, batch_size):
        max_idx = self.capacity if self.full else self.position
        indices = torch.randint(0, max_idx, (batch_size,), device=self.device)
        batch = [Transition(
            state=self.states[i].unsqueeze(0),
            action=self.actions[i].unsqueeze(0),
            next_state=self.next_states[i].unsqueeze(0),
            reward=self.rewards[i].unsqueeze(0),
            done=self.dones[i].unsqueeze(0)
        ) for i in indices]
        return batch

    def __len__(self):
        return self.capacity if self.full else self.position
