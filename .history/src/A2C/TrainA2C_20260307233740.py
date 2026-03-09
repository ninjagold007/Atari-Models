class A2CTrainer:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        # Initialize other A2C-specific parameters here

    
    
    def evaluate(self, envs, actor, num_episodes):
        total_rewards = []
        for episode in range(num_episodes):
            states = envs.reset()
            done = [False] * self.num_envs
            episode_rewards = 0

            while not all(done):
                actions = actor.choose_actions(states)
                next_states, rewards, done, _ = envs.step(actions)
                episode_rewards += sum(rewards)
                states = next_states

            total_rewards.append(episode_rewards)

        average_reward = sum(total_rewards) / num_episodes
        return average_reward
        

# Actor (πθ)
#    ↓ chooses
# Action a_t
#    ↓ causes
# Reward r_t, State s_{t+1}
#    ↓ used by
# Critic (Vφ)
#    ↓ computes
# Advantage A_t
#    ↓ updates
# Actor (πθ)
