import numpy as np
import torch
from matplotlib import pylab as plt
import random
import sumo_rl
import traci  # Required for direct vehicle checks
from ql_agent import QlAgent

# Define networks
network_files = ['../nets/road1.net.xml', '../nets/road2.net.xml', '../nets/road3.net.xml']
route_files = ['../nets/road1.rou.xml', '../nets/road2.rou.xml', '../nets/road3.rou.xml']

# Q-learning parameters
epochs = 5
epsilon = 1
gamma = 0.9

# Function to detect emergency vehicles
def check_emergency_vehicle(ts_id):
    """Returns the count of emergency vehicles detected in lanes controlled by the given traffic signal."""
    emergency_count = 0
    try:
        lanes = traci.trafficlight.getControlledLanes(ts_id)  # Get controlled lanes
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh in vehicles:
                if traci.vehicle.getTypeID(veh) == "DEFAULT_CONTAINERTYPE":  # Emergency vehicle ID
                    emergency_count += 1
    except Exception as e:
        print(f"Error checking emergency vehicle at {ts_id}: {e}")
    return emergency_count  # Return count instead of True/False

# Initialize model
ql_agent = None

# Training across all roads
avg_rewards = []

for epoch in range(epochs):
    for network_file, route_file in zip(network_files, route_files):
        print(f"🔍 Checking traffic signals in {network_file} with {route_file}")

        env = sumo_rl.SumoEnvironment(
            net_file=network_file,
            route_file=route_file,
            use_gui=False,
            num_seconds=20000,  # Ensure SUMO runs long enough
            single_agent=False
        )

        observations = env.reset()
        traffic_signals = list(env.traffic_signals.keys())

        if not traffic_signals:
            print(f"⚠️ Warning: No traffic signals found in {network_file}!")
        else:
            print(f"✅ Traffic signals found in {network_file}: {traffic_signals}")

        # Initialize the Q-learning agent only once, based on first traffic signal
        if ql_agent is None:
            first_ts = list(env.traffic_signals.keys())[0]
            input_shape = len(observations[first_ts]) + 1  # +1 for emergency detection
            ql_agent = QlAgent(input_shape=input_shape)

        done = {"__all__": False}
        step_count = 0
        input_data = {}

        while not done["__all__"]:
            step_count += 1

            for ts_id in env.traffic_signals.keys():  # Iterate over all traffic signals
                # Check for emergency vehicles
                emergency_count = check_emergency_vehicle(ts_id)

                # Ensure the traffic signal exists in observations
                if ts_id not in observations:
                    print(f"Warning: Traffic signal {ts_id} not found in observations.")
                    continue  # Skip if not found

                # Add emergency vehicle status to observation
                obs_with_emergency = np.append(observations[ts_id], emergency_count)
                input_data[ts_id] = torch.Tensor(obs_with_emergency).to(torch.float)

                # Select an action
                try:
                    pred_rewards = ql_agent.predict_rewards(input_data[ts_id])
                except Exception as e:
                    print(f"Error predicting rewards for {ts_id}: {e}")
                    continue  # Skip this signal if an error occurs

                # Get valid transitions
                current_phase = env.traffic_signals[ts_id].green_phase
                valid_transitions = [key for key in env.traffic_signals[ts_id].yellow_dict.keys() if key[0] == current_phase]
                action = random.choice(valid_transitions)[1] if valid_transitions else current_phase

                # Modify action if emergency vehicles are detected
                if emergency_count > 0:
                    if action != env.traffic_signals[ts_id].green_phase:
                        action = env.traffic_signals[ts_id].green_phase
                        print(f"🚨 Emergency at {ts_id}, switching to green!")

                # Step in SUMO environment
                observations, rewards, done, infos = env.step({ts_id: action})

                # Modify rewards based on emergency vehicles
                if emergency_count > 0:
                    if action == env.traffic_signals[ts_id].green_phase:
                        rewards[ts_id] += 10 * emergency_count  # Reward for allowing emergency vehicles
                    else:
                        rewards[ts_id] -= 10 * emergency_count  # Penalty for blocking

                # Prepare next input data
                new_obs_with_emergency = np.append(observations[ts_id], emergency_count)
                input_data2 = torch.Tensor(new_obs_with_emergency).to(torch.float)

                # Compute Q-learning update
                with torch.no_grad():
                    q_reward = rewards[ts_id] + gamma * torch.max(ql_agent.predict_rewards(input_data2))

                avg_reward = rewards[ts_id]
                avg_rewards.append(avg_reward)
                print(f'Epoch {epoch+1}, Step {step_count}, Traffic Signal {ts_id}: Reward = {avg_reward}')

                # Train only if no emergency vehicle is present
                if emergency_count == 0:
                    ql_agent.learn(pred_rewards[action].clone().detach().requires_grad_(True),
                                   torch.tensor([q_reward], dtype=torch.float32),
                                   emergency_count)

                # Update input data
                input_data[ts_id] = input_data2

            # Stop after a certain number of steps
            if step_count > 100:
                break

        env.close()

# Save the trained model
model_path = "../trained_models/ql_model.pth"
torch.save(ql_agent.model.state_dict(), model_path)

# Plot training rewards
fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Training Steps")
ax.set_ylabel("Reward")
fig.set_size_inches(9, 5)
ax.scatter(np.arange(len(avg_rewards)), avg_rewards)
plt.title("Training Rewards for Combined Model")
plt.show()
