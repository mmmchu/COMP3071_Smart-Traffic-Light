import numpy as np
import torch
import random
import sumo_rl
import traci  # Required for direct vehicle checks
from agents.ql_agent import QlAgent
import matplotlib.pyplot as plt

# Define networks
network_files = ['nets/road1.net.xml', 'nets/road2.net.xml', 'nets/road3.net.xml']
route_files = ['nets/road1.rou.xml', 'nets/road2.rou.xml', 'nets/road3.rou.xml']

# Load the trained model
model_path = "trained_models/ql_model.pth"

# Initialize SUMO environment (to determine input size)
env = sumo_rl.SumoEnvironment(
    net_file=network_files[0],
    route_file=route_files[0],
    use_gui=True,
    num_seconds=20000,
    single_agent=False
)

observations = env.reset()

# **Get input shape (+1 for emergency vehicle detection)**
first_ts = list(env.traffic_signals.keys())[0]
input_shape = len(observations[first_ts]) + 1
ql_agent = QlAgent(input_shape=input_shape)

# **Load trained model weights**
ql_agent.model.load_state_dict(torch.load(model_path))
ql_agent.model.eval()  # Set model to evaluation mode

# Function to detect emergency vehicles at a traffic signal
def check_emergency_vehicle(ts_id):
    """Returns the number of emergency vehicles detected in any lane controlled by the given traffic signal."""
    try:
        lanes = traci.trafficlight.getControlledLanes(ts_id)
        count = sum(1 for lane in lanes for veh in traci.lane.getLastStepVehicleIDs(lane)
                    if traci.vehicle.getTypeID(veh) == "DEFAULT_CONTAINERTYPE")
        return count  # Return emergency vehicle count
    except Exception as e:
        print(f"Error checking emergency vehicle: {e}")
    return 0  # Default to 0 if error occurs


# **Evaluation across all networks**
avg_rewards = []

for network_file, route_file in zip(network_files, route_files):
    print(f"🚦 Evaluating on {network_file} with {route_file}")

    env = sumo_rl.SumoEnvironment(
        net_file=network_file,
        route_file=route_file,
        use_gui=True,
        num_seconds=20000,
        single_agent=False
    )

    observations = env.reset()
    done = {"__all__": False}
    step_count = 0

    while not done["__all__"]:
        step_count += 1

        for ts_id in env.traffic_signals.keys():  # Evaluate all traffic signals
            emergency_count = check_emergency_vehicle(ts_id)
            obs_with_emergency = np.append(observations[ts_id], emergency_count)
            input_data = torch.Tensor(obs_with_emergency).to(torch.float)

            # **Handle emergency vehicle priority**
            if emergency_count > 0:
                current_phase = env.traffic_signals[ts_id].green_phase
                valid_transitions = [key[1] for key in env.traffic_signals[ts_id].yellow_dict.keys() if key[0] == current_phase]

                # **Prioritize the lane with the most emergency vehicles**
                best_emergency_action = None
                max_emergency_lane_count = 0

                for valid_action in valid_transitions:
                    lanes = traci.trafficlight.getControlledLanes(ts_id)
                    lane_emergency_counts = [sum(1 for veh in traci.lane.getLastStepVehicleIDs(lane)
                                                 if traci.vehicle.getTypeID(veh) == "DEFAULT_CONTAINERTYPE")
                                             for lane in lanes]
                    if sum(lane_emergency_counts) > max_emergency_lane_count:
                        max_emergency_lane_count = sum(lane_emergency_counts)
                        best_emergency_action = valid_action

                action = best_emergency_action if best_emergency_action else random.choice(valid_transitions) if valid_transitions else current_phase
                print(f"🚨 Emergency at {ts_id}: Switching to phase {action}!")

            else:
                # Predict Q-values and select the best action
                with torch.no_grad():
                    pred_rewards = ql_agent.predict_rewards(input_data)

                current_phase = env.traffic_signals[ts_id].green_phase
                best_action = torch.argmax(pred_rewards).item()

                # Ensure the selected action is valid
                valid_transitions = [key[1] for key in env.traffic_signals[ts_id].yellow_dict.keys() if key[0] == current_phase]
                if best_action not in valid_transitions:
                    print(f"⚠️ Invalid action {best_action} for {ts_id}. Selecting a valid one.")
                    action = random.choice(valid_transitions) if valid_transitions else current_phase
                else:
                    action = best_action

            # Step in the SUMO environment
            observations, rewards, done, infos = env.step({ts_id: action})

            # **Modify rewards for emergency vehicles**
            if emergency_count > 0:
                if action == env.traffic_signals[ts_id].green_phase:
                    rewards[ts_id] += 10 * emergency_count  # Reward for allowing emergency vehicles
                else:
                    rewards[ts_id] -= 10 * emergency_count  # Penalty for blocking

            avg_rewards.append(rewards[ts_id])
            print(f"Step {step_count}, Signal {ts_id}: Reward = {rewards[ts_id]}")

        # Stop after a certain number of steps
        if step_count > 500:
            break

    env.close()

# **Plot evaluation rewards**
plt.figure(figsize=(9, 5))
plt.xlabel("Evaluation Steps")
plt.ylabel("Reward")
plt.scatter(np.arange(len(avg_rewards)), avg_rewards, color='b', alpha=0.5)
plt.title("Evaluation Rewards for Trained Q-Learning Model")
plt.show()
