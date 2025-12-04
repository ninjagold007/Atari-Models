import torch.nn as nn
import torch.nn.functional as F
class PPOModel(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(PPOModel, self).__init__()
        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # compute conv output size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,8,4),4,2),3,1)
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,8,4),4,2),3,1)
        linear_input_size = conv_h * conv_w * 64

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(512, n_actions)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0  # normalize pixel values
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        policy_logits = self.policy_head(x)
        state_values = self.value_head(x)
        return policy_logits, state_values