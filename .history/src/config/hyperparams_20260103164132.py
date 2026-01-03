from pathlib import Path

class params:
    def __init__(self):
        
        #GAME ENVIRONMENT
        self.ENV_ID = "ALE/Breakout-v5"
        # self.ENV_ID = "Pong-v5"
        # self.ENV_ID = "CartPole-v1"
        # self.ENV_ID = "ALE/SpaceInvaders-v5"

        #MODEL
        self.MODEL_TYPE = "DQN"  
        # self.MODEL_TYPE = "A2C"
        # self.MODEL_TYPE = "PPO"
        # self.MODEL_TYPE = "RainbowDQN"


        #HYPERPARAMETERS
        self.NUM_ENVS = 8
        self.FRAME_H, self.FRAME_W = 84, 84
        self.NUM_STACK = 4
        self.FRAME_SKIP = 4
        self.BATCH_SIZE = 16
        self.GAMMA = 0.99
        self.FRAME_W = 84
        self.FRAME_H = 84

        self.EPS_START = 1.0
        self.EPS_END = 0.05
        self.EPS_DECAY = 500_000  # slower decay for exploration

        self.LR = 1e-4
        self.TARGET_TAU = 0.01    # soft update
        self.REPLAY_CAPACITY = 10_000
        self.LEARN_START = 2000
        self.OPTIMIZE_EVERY = 2

        self.MAX_EPISODE_STEPS = 2000
        self.SAVE_DIR = "SavedWeights"
        WEIGHTS = Path(__file__).resolve().parents[1] / "SavedWeights" / "policy_net_best_DQN.pth"
        self.SAVE_EVERY_FINISHED = 100
        self.TOTAL_FINISHED_EPISODES_TO_RUN = 4000