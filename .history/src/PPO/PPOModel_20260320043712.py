import torch
import torch.nn as nn


class PPO(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        
        # Convolutional layers
        c, h, w = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU()
        )

        # Calculate conv output size
        def conv_out(x, k, s): 
            return (x - (k - 1) - 1)//s + 1

        conv_h = conv_out(conv_out(conv_out(h,8,4),4,2),3,1)
        conv_w = conv_out(conv_out(conv_out(w,8,4),4,2),3,1)
        linear_input = conv_h * conv_w * 64

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(linear_input, 512),
            nn.ReLU(),
        )

        # Output layers for policy and value
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    # moves data forward through the network
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value

    # sample action + log_prob + value (used during rollout)
    def get_action_and_value(self, state):
        logits, value = self.forward(state)

        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value.squeeze(-1)

    # evaluate actions from buffer (used during training)
    def evaluate_actions(self, states, actions):
        logits, value = self.forward(states)

        dist = torch.distributions.Categorical(logits=logits)

        # make sure shapes match
        if actions.dim() > 1:
            actions = actions.squeeze(-1)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, value.squeeze(-1), entropy