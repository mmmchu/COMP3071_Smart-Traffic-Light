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
epochs = 5
epsilon = 1
gamma = 0.9

# Initialize SUMO environment
env = sumo_rl.SumoEnvironment(
    net_file=network_files[0],  # Use the first network file to determine input size
    route_file=route_files[0],
    use_gui=False,
    num_seconds=20000,
    single_agent=False
)

observations = env.reset()

# Get the actual input shape dynamically from the first traffic signal
first_ts = list(env.traffic_signals.keys())[0]  # Pick the first traffic signal
input_shape = len(observations[first_ts])  # Get observation size

# Now, initialize the Q-learning agent correctly
ql_agent = QlAgent(input_shape=input_shape)


# Training across both roads
avg_rewards = []

for epoch in range(epochs):
    for network_file, route_file in zip(network_files, route_files):
        print(f"Training on {network_file} with {route_file}")

        # Create environment
        env = sumo_rl.SumoEnvironment(
            net_file=network_file,
            route_file=route_file,
            use_gui=False,  # Faster training
            num_seconds=20000,
            single_agent=False
        )

        observations = env.reset()

        # Select one traffic signal randomly per training step
        target_ts = random.choice(list(env.traffic_signals.keys()))

        done = {"__all__": False}
        i = 0
        input_data = {}

        while not done["__all__"]:
            i += 1

            # Prepare input data (No neighbors)
            input_data[target_ts] = torch.Tensor(observations[target_ts]).to(torch.float)

            # Predict rewards
            try:
                pred_rewards = ql_agent.predict_rewards(input_data[target_ts])
            except Exception as e:
                print(f"Error in predict_rewards for {target_ts}: {e}")
                break

            # Select action
            current_phase = env.traffic_signals[target_ts].green_phase
            valid_transitions = [key for key in env.traffic_signals[target_ts].yellow_dict.keys() if key[0] == current_phase]
            action = random.choice(valid_transitions)[1] if valid_transitions else current_phase

            # Step environment
            observations, rewards, done, infos = env.step({target_ts: action})

            # Prepare new input data (No neighbors)
            input_data2 = torch.Tensor(observations[target_ts]).to(torch.float)

            # Compute Q-learning update
            with torch.no_grad():
                q_reward = rewards[target_ts] + gamma * torch.max(ql_agent.predict_rewards(input_data2))

            # Log average reward
            avg_reward = rewards[target_ts]
            avg_rewards.append(avg_reward)
            print(f'Epoch {epoch+1}, Step {i}: Reward = {avg_reward}')

            if i > 100:
                break

            # Train agent
            ql_agent.learn(pred_rewards[action].clone().detach().requires_grad_(True),
                           torch.tensor([q_reward], dtype=torch.float32))

            # Decay epsilon
            if epsilon > 0.1:
                epsilon -= 1 / 1500

            input_data[target_ts] = input_data2

        env.close()

# Save **one single combined model** for both roads
combined_model_path = "trained_models/ql_model.pth"
torch.save(ql_agent.model.state_dict(), combined_model_path)

# Plot training rewards
fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Training Steps")
ax.set_ylabel("Reward")
fig.set_size_inches(9, 5)
ax.scatter(np.arange(len(avg_rewards)), avg_rewards)
plt.title("Training Rewards for Combined Model")
plt.show()
