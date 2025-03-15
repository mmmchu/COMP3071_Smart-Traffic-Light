import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, lr=0.001, batch_size=32, memory_size=1000):
        self.state_size = state_size  # 3 priority levels
        self.action_size = action_size  # Number of signals
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=0.1):
        if random.uniform(0, 1) < epsilon:
            return random.randrange(self.action_size)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()  # Exploit

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = self.model(state).detach()
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            output = self.model(state)
            loss = F.mse_loss(output[action], target[action])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, episodes):
        for _ in range(episodes):
            self.replay()
            self.update_target_model()

            # Example Usage
            if __name__ == "__main__":
                agent = DQNAgent(state_size=3, action_size=4)  # 3 priority states, 4 traffic signals

                # Simulated training loop
                for episode in range(100):
                    state = np.random.choice([0, 1, 2])  # Random traffic priority
                    action = agent.act([state])  # Choose action
                    reward = random.randint(-5, 10)  # Simulated reward
                    next_state = np.random.choice([0, 1, 2])  # Next priority state
                    done = episode == 99  # End condition
                    agent.remember([state], action, reward, [next_state], done)
                    agent.train(1)

                print("Training complete!")
