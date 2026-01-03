import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import time
import torch
import gymnasium as gym
import ale_py
from collections import deque
from DQN.DQNModel import DQN
from Preproccessing.Preproccessing import preprocess_frame
from DQN.ReplayBuffer import ReplayBuffer
from config.hyperparams import params
hp = params()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_STACK = hp.NUM_STACK
FRAME_H, FRAME_W = hp.FRAME_H, hp.FRAME_W

gym.register_envs(ale_py)




def init_stack(first_frame):
    """Stack = 4 identical frames, matching training."""
    f = first_frame.clone()
    d = deque([f, f.clone(), f.clone(), f.clone()], maxlen=NUM_STACK)
    s = torch.cat(list(d), dim=0).unsqueeze(0)
    return s, d




class load:
    def __init__(self):
        self.env = gym.make("ALE/Breakout-v5", render_mode="human")

        n_actions = self.env.action_space.n
        self.model = DQN((NUM_STACK, FRAME_H, FRAME_W), n_actions).to(device)
        self.model.load_state_dict(torch.load("C:\\Users\\ninja\\School\\Atari-Models\\src\\SavedWeights\\policy_net_best_DQN.pth", map_location=device))
        self.model.eval()

    def run(self, episodes=3, delay=0.01):
        try:
            for ep in range(episodes):

                # --------------------------
                # RESET
                # --------------------------
                frame, _ = self.env.reset()
                frame_t = preprocess_frame(frame)

                # initial stack = 4 identical frames (training behavior)
                state, stacked = init_stack(frame_t)

                # FIRE once to start ball
                frame, _, _, _, _ = self.env.step(1)
                frame_t = preprocess_frame(frame)
                stacked.append(frame_t)
                state = torch.cat(list(stacked), dim=0).unsqueeze(0)

                ep_reward = 0
                done = False
                step = 0

                while not done:
                    step += 1

                    # --------------------------
                    # GREEDY ACTION
                    # --------------------------
                    with torch.no_grad():
                        q_vals = self.model(state.to(device))
                        action = q_vals.argmax(1).item()

                    # ----- DEBUG: print Q-values and chosen action -----
                    print(f"Step {step} | Q-values: {q_vals.cpu().numpy()} | Chosen action: {action}")

                    # --------------------------
                    # ENV STEP
                    # --------------------------
                    frame, reward, terminated, truncated, _ = self.env.step(action)
                    ep_reward += reward

                    # stack frames
                    frame_t = preprocess_frame(frame)
                    stacked.append(frame_t)
                    state = torch.cat(list(stacked), dim=0).unsqueeze(0)

                    done = terminated or truncated

                    # slow slightly for visibility
                    time.sleep(delay)

                print(f"Episode {ep+1} reward: {ep_reward}\n")

        finally:
            self.env.close()

loader = load()
loader.run()
