import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define the DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return np.argmax(action_values.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(torch.FloatTensor(next_state).unsqueeze(0))).item()

            target_f = self.model(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(torch.FloatTensor(state).unsqueeze(0)), torch.FloatTensor(target_f))
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Traffic Environment for Training
class TrafficEnv:
    def __init__(self):
        self.state_size = 4  # Number of signals (each with vehicle count)
        self.action_size = 4  # Which signal to turn green next
        self.reset()

    def reset(self):
        self.vehicle_counts = [random.randint(0, 10) for _ in range(4)]
        return np.array(self.vehicle_counts, dtype=np.float32)

    def step(self, action):
        reward = -sum(self.vehicle_counts)  # Reward is minimizing total waiting vehicles
        self.vehicle_counts[action] = max(0, self.vehicle_counts[action] - random.randint(1, 5))
        self.vehicle_counts = [count + random.randint(0, 3) for count in
                               self.vehicle_counts]  # Vehicles arrive randomly
        next_state = np.array(self.vehicle_counts, dtype=np.float32)
        done = False
        return next_state, reward, done


# Training Loop
if __name__ == "__main__":
    env = TrafficEnv()
    agent = DQNAgent(state_size=4, action_size=4)
    episodes = 1000

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay()
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    torch.save(agent.model.state_dict(), "dqn_traffic_model.pth")
