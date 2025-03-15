import numpy as np
import torch
from matplotlib import pylab as plt
import random
import sumo_rl

from agents.ql_agent import QlAgent  # Ensure this path is correct

# Define networks
network_files = ['nets/road1.net.xml', 'nets/road2.net.xml']
route_files = ['nets/road1.rou.xml', 'nets/road2.rou.xml']

# Q-learning parameters
gamma = 0.9

# Initialize SUMO environment
env = sumo_rl.SumoEnvironment(
    net_file=network_files[0],  # Use the first network file to determine input size
    route_file=route_files[0],
    use_gui=True,  # Set to True to visualize the simulation
    num_seconds=20000,
    single_agent=False
)

observations = env.reset()

# Get the actual input shape dynamically from the first traffic signal
first_ts = list(env.traffic_signals.keys())[0]  # Pick the first traffic signal
input_shape = len(observations[first_ts])  # Get observation size

# Initialize the Q-learning agent
ql_agent = QlAgent(input_shape=input_shape)

# Load the trained model
combined_model_path = "trained_models/ql_model.pth"
ql_agent.model.load_state_dict(torch.load(combined_model_path))
ql_agent.model.eval()  # Set the model to evaluation mode

# Run simulation
avg_rewards = {network_file: [] for network_file in network_files}  # Track rewards for each road

for network_file, route_file in zip(network_files, route_files):
    print(f"Running simulation on {network_file} with {route_file}")

    # Create environment
    env = sumo_rl.SumoEnvironment(
        net_file=network_file,
        route_file=route_file,
        use_gui=True,  # Set to True to visualize the simulation
        num_seconds=20000,
        single_agent=False
    )

    observations = env.reset()

    # Select one traffic signal randomly per simulation step
    target_ts = random.choice(list(env.traffic_signals.keys()))

    done = {"__all__": False}
    i = 0
    input_data = {}

    while not done["__all__"]:
        i += 1

        # Prepare input data (No neighbors)
        input_data[target_ts] = torch.Tensor(observations[target_ts]).to(torch.float)

        # Predict rewards
        with torch.no_grad():
            pred_rewards = ql_agent.predict_rewards(input_data[target_ts])

        # Select action
        current_phase = env.traffic_signals[target_ts].green_phase
        valid_transitions = [key for key in env.traffic_signals[target_ts].yellow_dict.keys() if key[0] == current_phase]
        action = random.choice(valid_transitions)[1] if valid_transitions else current_phase

        # Step environment
        observations, rewards, done, infos = env.step({target_ts: action})

        # Log average reward
        avg_reward = rewards[target_ts]
        avg_rewards[network_file].append(avg_reward)
        print(f'Step {i}: Reward = {avg_reward}')

        if i > 100:
            break

        input_data[target_ts] = torch.Tensor(observations[target_ts]).to(torch.float)

    env.close()

# Plot simulation rewards for each road
fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Simulation Steps")
ax.set_ylabel("Reward")
fig.set_size_inches(9, 5)

# Define colors for each road
colors = ['b', 'g', 'r', 'c', 'm', 'y']

for idx, (network_file, rewards) in enumerate(avg_rewards.items()):
    ax.scatter(np.arange(len(rewards)), rewards, color=colors[idx % len(colors)], label=network_file)

plt.title("Simulation Rewards for Combined Model")
plt.legend()  # Add a legend to distinguish roads
plt.show()