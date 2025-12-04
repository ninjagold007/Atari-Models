from config.hyperparams import params
hp = params()
from DQN.Train_DQN import DQNTrainer
from A2C.TrainA2C import A2CTrainer
from PPO.Train_PPO import PPOTrainer
from RainbowDQN.Train_RainbowDQN import RainbowDQNTrainer

# Initialize and run the trainer
model = hp.MODEL_TYPE
if(model == "DQN"):
    trainer = DQNTrainer(num_envs=hp.NUM_ENVS)

elif(hp.MODEL_TYPE == "A2C"):
    trainer = A2CTrainer(num_envs=hp.NUM_ENVS)

elif(hp.MODEL_TYPE == "PPO"):
    trainer = PPOTrainer(num_envs=hp.NUM_ENVS)

elif(hp.MODEL_TYPE == "RainbowDQN"):
    trainer = RainbowDQNTrainer(num_envs=hp.NUM_ENVS)

trainer.run()