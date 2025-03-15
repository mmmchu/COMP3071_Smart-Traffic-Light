import torch
from DQN import DQNAgent  # Import your updated DQNAgent class

# Initialize the agent (must match the original state and action size)
state_size = 13
action_size = 4
agent = DQNAgent(state_size, action_size)

# Load the trained model
agent.load_model("trained_model_traffic_0.5.zip")

# Example: Use the trained model to choose an action
state = [0.2, 0.5, 0.1, 0.4, 0.3, 0.7, 0.8, 0.2, 1.0, 0.6, 0.3, 0.2, 0.5]
action = agent.act(state)

print(f"Chosen action: {action}")
