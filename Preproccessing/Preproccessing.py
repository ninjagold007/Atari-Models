# --------------------------
# Preprocessing helpers
# --------------------------
import cv2
import torch
from config.hyperparams import params
hp = params()  # create an instance

# Preprocess a single frame by converting to grayscale, resizing, and normalizing
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (hp.FRAME_W, hp.FRAME_H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized).to(dtype=torch.float32) / 255.0
    return t.unsqueeze(0)  # 1 x H x W