import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sumo_rl
import os
import numpy as np

# Actor-Critic network (Adaptive to different action spaces)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, max_action_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, max_action_dim),  # Use max action dimension
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

# PPO Algorithm
class PPO:
    def __init__(self, state_dim, max_action_dim, hidden_size, lr, gamma, clip_ratio, K_epoch):
        self.actor_critic = ActorCritic(state_dim, max_action_dim, hidden_size)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.K_epoch = K_epoch

    def choose_action(self, state, action_dim):
        """ Selects action while ensuring compatibility with varying action spaces. """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_logits, _ = self.actor_critic(state)
        action_probs = F.softmax(action_logits[:, :action_dim], dim=-1)  # Use only available actions
        action = torch.multinomial(action_probs, 1).item()  # Sample action
        return action, action_probs[0, action].item()

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns for stability
        return (returns - returns.mean()) / (returns.std() + 1e-5)

    def update(self, states, actions, old_probs, rewards):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_probs = torch.FloatTensor(old_probs)
        returns = self.compute_returns(rewards)

        for _ in range(self.K_epoch):
            action_logits, values = self.actor_critic(states)
            log_probs = F.log_softmax(action_logits, dim=-1)  # Log softmax for stability
            new_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

            critic_loss = F.mse_loss(values.squeeze(), returns)

            ratio = torch.exp(new_probs - old_probs)  # PPO ratio
            surr1 = ratio * returns
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * returns
            actor_loss = -torch.min(surr1, surr2).mean()

            loss = actor_loss + critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def pad_state(state, max_state_dim):
    """ Pads smaller states with zeros to match max_state_dim. """
    padded_state = np.zeros(max_state_dim, dtype=np.float32)
    padded_state[:len(state)] = state
    return padded_state

def train(roads=["road1", "road2"], hidden_size=64, lr=3e-4, gamma=0.99, clip_ratio=0.2, K_epoch=10, max_steps=100, max_episodes=100):
    max_state_dim = 0
    max_action_dim = 0

    # Determine the max state & action dimensions across all roads
    for road in roads:
        env = sumo_rl.SumoEnvironment(
            net_file=f"C:/Users/mabel/PycharmProjects/smart_traffic_light/nets/{road}.net.xml",
            route_file=f"C:/Users/mabel/PycharmProjects/smart_traffic_light/nets/{road}.rou.xml",
            use_gui=False,
            num_seconds=1000,
            single_agent=True
        )
        max_state_dim = max(max_state_dim, env.observation_space.shape[0])
        max_action_dim = max(max_action_dim, env.action_space.n)
        env.close()

    print(f"Max State Dim: {max_state_dim}, Max Action Dim: {max_action_dim}")

    # Initialize model with max dimensions
    agent = PPO(max_state_dim, max_action_dim, hidden_size, lr, gamma, clip_ratio, K_epoch)

    for road in roads:
        print(f"Training on {road}...")

        env = sumo_rl.SumoEnvironment(
            net_file=f"C:/Users/mabel/PycharmProjects/smart_traffic_light/nets/{road}.net.xml",
            route_file=f"C:/Users/mabel/PycharmProjects/smart_traffic_light/nets/{road}.rou.xml",
            use_gui=False,
            num_seconds=1000,
            single_agent=True
        )

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n  # Get actual action space per road

        for episode in range(max_episodes):
            state = pad_state(env.reset()[0], max_state_dim)
            done = False
            total_reward = 0

            states, actions, old_probs, rewards = [], [], [], []

            for _ in range(max_steps):
                action, action_prob = agent.choose_action(state, action_dim)
                next_state, reward, done, _, _ = env.step(action)

                states.append(state)
                actions.append(action)
                old_probs.append(np.log(action_prob + 1e-5))  # Store log probabilities

                rewards.append(reward)

                state = pad_state(next_state, max_state_dim)
                total_reward += reward
                if done:
                    break

            agent.update(states, actions, old_probs, rewards)
            print(f"Road: {road} | Episode {episode + 1}: Total Reward = {total_reward}")

        env.close()

    # Ensure the trained_models directory exists
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "trained_models"))
    os.makedirs(save_dir, exist_ok=True)

    # Save the trained model
    save_path = os.path.join(save_dir, "ppo_model.pth")
    torch.save(agent.actor_critic.state_dict(), save_path)
    print(f"Model saved at {save_path}.")

if __name__ == "__main__":
    train()
