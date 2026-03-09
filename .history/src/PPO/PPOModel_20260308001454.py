import torch
import torch.nn as nn

# --------------------------
# PPO model
# --------------------------
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
        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    #fc means fully connected
    # moves data forward through the network
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x =self.fc(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
