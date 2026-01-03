import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import ale_py
import torch
from collections import deque
from DQN.DQNModel import DQN
from Preproccessing.Preproccessing import preprocess_frame
from config.hyperparams import params

hp = params()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gym.register_envs(ale_py)


class load:
    def __init__(self):
        pass

    def run(self):
        env = gym.make("ALE/Breakout-v5", render_mode="human")
        actions = env.action_space.n

        model = DQN((hp.NUM_STACK, hp.FRAME_H, hp.FRAME_W), actions).to(device)
        model.load_state_dict(torch.load(
            r"C:\Users\ninja\School\Atari-Models\src\SavedWeights\policy_net_best_DQN.pth",
            map_location=device
        ))
        model.eval()

        # ---- init state (same as training) ----
        frame, _ = env.reset()
        pf = preprocess_frame(frame).to(device).float()
        stacked_frames = deque([pf] * hp.NUM_STACK, maxlen=hp.NUM_STACK)
        state = torch.cat(list(stacked_frames), dim=0).unsqueeze(0)

        # ---- play loop ----
        while True:
            with torch.no_grad():
                action = model(state).argmax(dim=1).item()

            frame, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            pf = preprocess_frame(frame).to(device).float()
            stacked_frames.append(pf)

            if done:
                frame, _ = env.reset()
                pf = preprocess_frame(frame).to(device).float()
                stacked_frames = deque([pf] * hp.NUM_STACK, maxlen=hp.NUM_STACK)

            state = torch.cat(list(stacked_frames), dim=0).unsqueeze(0)


loader = load()
loader.run()
