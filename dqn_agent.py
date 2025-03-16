import numpy as np
import random
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import TensorBoard
import time
from traffic_environment import TrafficEnvironment  # Import TrafficEnvironment
import logging


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  
        q_values = self.model.predict(np.array([state]))[0]
        return np.argmax(q_values)  

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(np.array([next_state]))[0])
            target_f = self.model.predict(np.array([state]))[0]
            target_f[action] = target
            self.model.fit(np.array([state]), np.array([target_f]), epochs=1, verbose=0, callbacks=[self.tensorboard])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


episodes = 100  # Define the number of episodes
env = TrafficEnvironment(tls_id="cluster_4961708598_5420328763_5420328770_5800404149")  # Define env
agent = DQNAgent(state_size=4, action_size=2)  # Initialize agent

for episode in range(episodes):
    state = env.get_state()
    total_reward = 0
    total_wait_time = 0  # Track waiting time
    total_queue_length = 0
    steps = 0
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, wait_time, queue_length = env.step(action)  # Modify step() to return extra info
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        total_reward += reward
        total_wait_time += wait_time
        total_queue_length += queue_length
        steps += 1

    avg_wait_time = total_wait_time / steps
    avg_queue_length = total_queue_length / steps

    agent.train()
    
    # Log results
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Episode {episode}: Total Reward = {total_reward}, Avg Wait Time = {avg_wait_time}, Avg Queue = {avg_queue_length}")
    print(f"Episode {episode}: Reward={total_reward}, Avg Wait={avg_wait_time:.2f}, Avg Queue={avg_queue_length:.2f}")
