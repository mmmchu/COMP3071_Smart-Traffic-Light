import traci
from dqn_agent import DQNAgent
from traffic_environment import TrafficEnvironment
import logging

# Configure logging
logging.basicConfig(filename='train_dqn.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def start_sumo(config_file):
    sumoCmd = ["sumo-gui", "-c", config_file, "--step-length", "1"]
    try:
        traci.start(sumoCmd)
        print("SUMO started successfully.")
    except traci.exceptions.FatalTraCIError as e:
        print(f"Failed to start SUMO: {e}")
        logging.error(f"Failed to start SUMO: {e}")
        raise

import traci
from dqn_agent import DQNAgent
from traffic_environment import TrafficEnvironment

def train_agent(episodes=1000):
    traci.start(["sumo-gui", "-c", "c:/Users/user/COMP3071_Smart-Traffic-Light/osm.sumocfg", "--step-length", "1"])
    
    env = TrafficEnvironment(tls_id="cluster_4961708598_5420328763_5420328770_5800404149")
    agent = DQNAgent(state_size=4, action_size=2)

    for episode in range(episodes):
        state = env.get_state()  # ✅ Now `env` is properly defined
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.train()

    agent.model.save("dqn_traffic_light.h5")
    traci.close()

if __name__ == "__main__":
    train_agent()