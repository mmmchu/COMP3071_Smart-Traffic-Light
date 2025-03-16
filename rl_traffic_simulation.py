import traci
import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

# ---------------- RL Traffic Light Environment ----------------
class TrafficEnvironment:
    def __init__(self, tls_id):
        self.tls_id = tls_id
        self.state_size = 4  # Number of lanes (customizable)
        self.action_size = 2  # 0 = keep phase, 1 = switch phase
        self.last_action_time = time.time()

    def get_state(self):
        lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        return np.array([traci.lane.getLastStepVehicleNumber(lane) for lane in lanes])

    def step(self, action):
        if time.time() - self.last_action_time < 5:
            return self.get_state(), 0, False  

        if action == 1:
            traci.trafficlight.setPhase(self.tls_id, (traci.trafficlight.getPhase(self.tls_id) + 1) % 2)

        traci.simulationStep()
        self.last_action_time = time.time()

        reward = -sum(traci.lane.getLastStepHaltingNumber(lane) for lane in traci.trafficlight.getControlledLanes(self.tls_id))
        next_state = self.get_state()
        done = traci.simulation.getMinExpectedNumber() <= 0  
        return next_state, reward, done

# ---------------- RL Agent (Deep Q-Network) ----------------
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

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
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
            self.model.fit(np.array([state]), np.array([target_f]), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ---------------- Training the RL Agent ----------------
def train_agent(episodes=500):
    sumoCmd = ["sumo", "-c", "osm.sumocfg"]
    traci.start(sumoCmd)

    tls_id = traci.trafficlight.getIDList()[0]  
    env = TrafficEnvironment(tls_id)
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    for episode in range(episodes):
        state = env.get_state()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.train()
        print(f"Episode {episode}: Total Reward = {total_reward}")

    agent.model.save("dqn_traffic_light.h5")
    traci.close()

if __name__ == "__main__":
    train_agent()
