import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import sumo_rl
import traci
from ql_agent import QlAgent

network_files = ['../nets/road1.net.xml', '../nets/road2.net.xml', '../nets/road3.net.xml']
route_files = ['../nets/road1.rou.xml', '../nets/road2.rou.xml', '../nets/road3.rou.xml']

# Fixed max input shape (computed previously)
max_input_shape = 49

epochs = 10
epsilon = 1
gamma = 0.9


def check_emergency_vehicle(traffic_id):
    count_emergency = 0
    try:
        lanes = traci.trafficlight.getControlledLanes(traffic_id)
        for lane in lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for veh in vehicles:
                if traci.vehicle.getTypeID(veh) == "DEFAULT_CONTAINERTYPE":
                    count_emergency += 1
    except Exception as exception:
        print(f"Error checking emergency vehicle at {traffic_id}: {exception}")
    return count_emergency


ql_agent = QlAgent(input_shape=max_input_shape)
avg_rewards = []
reward_data_per_road = {road: [] for road in network_files}

for epoch in range(epochs):
    for network_file, route_file in zip(network_files, route_files):
        print(f"\n🔍 Training on {network_file} with {route_file}")

        env = sumo_rl.SumoEnvironment(
            net_file=network_file,
            route_file=route_file,
            use_gui=False,
            num_seconds=20000,
            single_agent=False
        )

        observations = env.reset()
        traffic_signals = list(env.traffic_signals.keys())

        if not traffic_signals:
            print(f"⚠️ No traffic signals found in {network_file}!")
            continue

        done = {"__all__": False}
        step_count = 0
        input_data = {}

        while not done["__all__"]:
            step_count += 1
            actions = {}

            for ts_id in traffic_signals:
                emergency_count = check_emergency_vehicle(ts_id)

                if ts_id not in observations:
                    continue

                obs = np.append(observations[ts_id], emergency_count)
                obs_padded = np.pad(obs, (0, max_input_shape - len(obs)))
                obs_tensor = torch.Tensor(obs_padded).float()
                input_data[ts_id] = obs_tensor

                try:
                    pred_rewards = ql_agent.predict_rewards(obs_tensor)
                except Exception as e:
                    print(f"Error predicting rewards for {ts_id}: {e}")
                    continue

                current_phase = env.traffic_signals[ts_id].green_phase
                valid_transitions = [key for key in env.traffic_signals[ts_id].yellow_dict.keys() if
                                     key[0] == current_phase]
                next_phase = random.choice(valid_transitions)[1] if valid_transitions else current_phase

                if emergency_count > 0 and next_phase != current_phase:
                    next_phase = current_phase
                    print(f"🚨 Emergency at {ts_id}, keeping current phase!")

                actions[ts_id] = next_phase

            observations, rewards, done, infos = env.step(actions)

            for ts_id in actions.keys():
                emergency_count = check_emergency_vehicle(ts_id)

                if ts_id not in observations:
                    continue

                if emergency_count > 0:
                    if actions[ts_id] == env.traffic_signals[ts_id].green_phase:
                        rewards[ts_id] += 10 * emergency_count
                    else:
                        rewards[ts_id] -= 10 * emergency_count

                new_obs = np.append(observations[ts_id], emergency_count)
                new_obs_padded = np.pad(new_obs, (0, max_input_shape - len(new_obs)))
                next_obs_tensor = torch.Tensor(new_obs_padded).float()

                with torch.no_grad():
                    q_target = rewards[ts_id] + gamma * torch.max(ql_agent.predict_rewards(next_obs_tensor))

                pred_rewards = ql_agent.predict_rewards(input_data[ts_id])
                action_index = torch.argmax(pred_rewards).item()

                avg_rewards.append(rewards[ts_id])
                reward_data_per_road[network_file].append(rewards[ts_id])

                if emergency_count == 0:
                    ql_agent.learn(pred_rewards[action_index].clone().detach().requires_grad_(True),
                                   torch.tensor([q_target], dtype=torch.float32),
                                   emergency_count)

                input_data[ts_id] = next_obs_tensor

            if step_count > 100:
                break

        env.close()

# Save the trained model
torch.save(ql_agent.model.state_dict(), "../trained_models/ql_model.pth")

# Plotting reward curves
colors = plt.colormaps.get_cmap('tab10')(range(len(reward_data_per_road)))
fig, axs = plt.subplots(len(reward_data_per_road), figsize=(10, 3.5 * len(reward_data_per_road)))
if len(reward_data_per_road) == 1:
    axs = [axs]

for idx, (road, rewards) in enumerate(reward_data_per_road.items()):
    x = np.arange(len(rewards))
    axs[idx].scatter(x, rewards, color=colors[idx], label=f"Road {road.split('/')[-1]}", s=10, alpha=0.8)
    axs[idx].set_xlabel("Training Steps", fontsize=11)
    axs[idx].set_ylabel("Reward", fontsize=11)
    axs[idx].set_title(f"Training Rewards for Road {road.split('/')[-1]}", fontsize=13)
    axs[idx].grid(True, linestyle='--', alpha=0.5)
    axs[idx].legend(loc="upper right")
    if idx == 2:
        axs[idx].set_ylim(-2, 5)

fig.tight_layout(pad=2.5)
plt.show()
