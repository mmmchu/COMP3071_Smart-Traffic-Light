import torch
import torch.nn as nn
import torch.nn.functional as F
import sumo_rl
import numpy as np
import os
import matplotlib.pyplot as plt

# Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

def pad_state(state, max_state_dim):
    """Pads smaller states with zeros to match max_state_dim."""
    padded_state = np.zeros(max_state_dim, dtype=np.float32)
    padded_state[:len(state)] = state
    return padded_state

def evaluate(roads=["road1", "road2"], hidden_size=64, max_steps=100):
    model_path = "trained_models/ppo_model.pth"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, please train the model first.")
        return

    max_state_dim = 0
    max_action_dim = 0

    # Determine the max state & action dimensions
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

    # Load model
    agent = ActorCritic(max_state_dim, max_action_dim, hidden_size)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()

    # Initialize a dictionary to store rewards for each road
    rewards_dict = {road: [] for road in roads}

    for i, road in enumerate(roads):
        print(f"Evaluating on {road}...")
        env = sumo_rl.SumoEnvironment(
            net_file=f"C:/Users/mabel/PycharmProjects/smart_traffic_light/nets/{road}.net.xml",
            route_file=f"C:/Users/mabel/PycharmProjects/smart_traffic_light/nets/{road}.rou.xml",
            use_gui=False,
            num_seconds=1000,
            single_agent=True
        )

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        state = pad_state(env.reset()[0], max_state_dim)
        done = False

        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_logits, _ = agent(state_tensor)
            action_probs = F.softmax(action_logits[:, :action_dim], dim=-1)  # Adapt to action space
            action = torch.argmax(action_probs, dim=-1).item()

            next_state, reward, done, _, _ = env.step(action)
            state = pad_state(next_state, max_state_dim)
            rewards_dict[road].append(reward)

            if done:
                break

        print(f"Road: {road} | Total Reward: {sum(rewards_dict[road])}")
        env.close()

    # Plot all results at once
    colors = ["b", "g", "r", "c", "m", "y"]  # Different colors for different roads
    fig, ax = plt.subplots()
    for i, (road, rewards) in enumerate(rewards_dict.items()):
        ax.scatter(range(len(rewards)), rewards, color=colors[i % len(colors)], label=road)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Evaluation Rewards for All Roads")
    ax.legend()
    plt.show()  # Show final plot

if __name__ == "__main__":
    evaluate()