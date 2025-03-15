import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import random
import os
from collections import deque

# Define the DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, batch_size=64, memory_size=5000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)

        # Exploration parameters
        self.epsilon = 1.0       # Start with full exploration
        self.epsilon_min = 0.05  # Minimum exploration
        self.epsilon_decay = 0.999  # Slower decay for better exploration

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.update_target_model()

    def update_target_model(self):
        """Copies weights from the model to the target model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Stores experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy action selection"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploit

    def replay(self):
        """Trains the DQN model using replay memory"""
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states).detach()

        target_q_values = q_values.clone()
        for i in range(len(batch)):
            target_q_values[i, actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * torch.max(next_q_values[i])

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load_model(self, model_path):
        """Loads the trained model from a file."""
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")

    def select_action(self, state):
        """Chooses the best action using the trained model."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
        with torch.no_grad():
            action_values = self.model(state)  # Forward pass through the network
        return torch.argmax(action_values).item()  # Return the best action

    def train(self, episodes, traffic_data):
        """Trains the DQN model using traffic data"""
        for episode in range(episodes):
            traffic_data.index = 0  # RESET the dataset index at the start of each episode
            state = traffic_data.get_state()

            if state is None:
                print(f"Error: No valid data for Episode {episode + 1}.")
                break  # Stop training if data is invalid

            done = False
            total_reward = 0
            step = 0

            print(f"Episode {episode + 1}/{episodes} - Training started...")

            while not done:
                action = self.act(state)
                next_state, reward, done = traffic_data.step(action)

                if next_state is None:  # Stop if data runs out
                    break

                self.remember(state, action, reward, next_state, done)
                state = next_state
                self.replay()

                total_reward += reward
                step += 1

                if step % 10 == 0:
                    print(f"  Step {step}: Action={action}, Reward={reward:.2f}, Total Reward={total_reward:.2f}")

            if episode % 10 == 0:
                self.update_target_model()

            # Decay epsilon after each episode
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            print(f"Episode {episode + 1} completed - Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")

        print("Training complete!")


# Define the Traffic Data Environment
class TrafficData:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.index = 0

    def get_state(self):
        """Returns the current state (13 features)"""
        if self.index >= len(self.data):
            return None
        return self.data.iloc[self.index].tolist()  # Extract all 13 features

    def step(self, action):
        """Simulates an action and returns next_state, reward, done."""
        self.index += 1
        if self.index >= len(self.data):
            return None, 0, True

        next_row = self.data.iloc[self.index]
        next_state = next_row.tolist()

        # Extract queue length, waiting time, and flow rate for chosen signal
        queue_length = next_row.iloc[action] / 10  # Normalize
        waiting_time = next_row.iloc[action + 4] / 10  # Normalize
        flow_rate = next_row.iloc[action + 8] / 5  # Normalize

        # Updated Reward function
        reward = 10 - 0.5 * queue_length - 0.3 * waiting_time + 1.5 * flow_rate

        # Debugging - Track reward values
        print(f"Action: {action}, Queue: {queue_length:.2f}, Wait: {waiting_time:.2f}, Flow: {flow_rate:.2f}, Reward: {reward:.2f}")

        return next_state, reward, False

# Load traffic data
folder_name = "traffic_logs"
filename = os.path.join(folder_name, "traffic_data_0.5.csv")

if os.path.exists(filename):
    traffic_data = TrafficData(filename)
    state_size = 13
    action_size = 4
    agent = DQNAgent(state_size, action_size)

    # If a trained model exists, load it instead of retraining
    model_path = "trained_model_traffic_0.5.zip"
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print("Model loaded, skipping training.")
    else:
        agent.train(episodes=100, traffic_data=traffic_data)
        torch.save(agent.model.state_dict(), model_path)
        print("Training completed and model saved!")
else:
    print(f"Error: {filename} not found!")

