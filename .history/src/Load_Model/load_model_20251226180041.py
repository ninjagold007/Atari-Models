import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gymnasium as gym
import ale_py
import torch
from DQN.DQNModel import DQN
from config.hyperparams import params
hp = params()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gym.register_envs(ale_py)



# class load:
#     def __init__(self):
#         pass

#     def run(self):
#         env = gym.make("ALE/Breakout-v5", render_mode="human")
#         height, width, channels = env.observation_space.shape
#         actions = env.action_space.n

#         model = DQN((hp.NUM_STACK, hp.FRAME_H, hp.FRAME_W), actions).to(device)
#         model.load_state_dict(torch.load('C:\\Users\\ninja\\School\\Atari-Models\\src\\SavedWeights\\policy_net_best_DQN.pth', map_location=device))
#         model.eval()  # Set to evaluation mode

class load:
    def __init__(self):
        pass

    def run(self):
        env = gym.make("ALE/Breakout-v5", render_mode="human")
        obs, _ = env.reset()

        actions = env.action_space.n
        model = DQN((hp.NUM_STACK, hp.FRAME_H, hp.FRAME_W), actions).to(device)
        model.load_state_dict(
            torch.load(
                r"C:\Users\ninja\School\Atari-Models\src\SavedWeights\policy_net_best_DQN.pth",
                map_location=device
            )
        )
        model.eval()

        frame_stack = deque(maxlen=hp.NUM_STACK)

        processed = preprocess(obs)
        for _ in range(hp.NUM_STACK):
            frame_stack.append(processed)

        done = False
        while not done:
            state = np.stack(frame_stack, axis=0)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                action = model(state).argmax(dim=1).item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            processed = preprocess(obs)
            frame_stack.append(processed)

        env.close()
 
loader = load()
loader.run()
