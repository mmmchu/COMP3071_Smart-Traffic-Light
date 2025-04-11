import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sumo_rl
import traci
from ql_agent import QlAgent

# === Test Environment ===
net_file = "../nets/road2.net.xml"
route_file = "../nets/road2.rou.xml"
model_path = "../trained_models/ql_model.pth"
max_input_shape = 49

# === Setup Environment ===
env = sumo_rl.SumoEnvironment(
    net_file=net_file,
    route_file=route_file,
    use_gui=False,
    num_seconds=5000,
    single_agent=False
)

observations = env.reset()
ts_ids = list(env.traffic_signals.keys())
print("Traffic signals:", ts_ids)

agent = QlAgent(input_shape=max_input_shape)
agent.model.load_state_dict(torch.load(model_path))
agent.model.eval()

def check_emergency(ts):
    count = 0
    for lane in traci.trafficlight.getControlledLanes(ts):
        for v in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getTypeID(v) == "DEFAULT_CONTAINERTYPE":
                count += 1
    return count

# === Tracking Metrics ===
rewards_per_step = []
avg_waiting_times = []
avg_queue_lengths = []
phase_usage = {ts: [] for ts in ts_ids}

# === Simulation Loop ===
done = {"__all__": False}
step = 0
while not done["__all__"]:
    actions = {}
    for ts in ts_ids:
        emergency = check_emergency(ts)
        obs = np.append(observations[ts], emergency)
        obs_padded = np.pad(obs, (0, max_input_shape - len(obs)))
        obs_tensor = torch.tensor(obs_padded, dtype=torch.float32)

        ts_obj = env.traffic_signals[ts]
        current_phase = ts_obj.green_phase
        valid_transitions = [new for (old, new) in ts_obj.yellow_dict.keys() if old == current_phase]

        pred = agent.predict_rewards(obs_tensor)
        masked_pred = torch.full_like(pred, float('-inf'))
        for phase in valid_transitions:
            masked_pred[phase] = pred[phase]

        action = torch.argmax(masked_pred).item()
        actions[ts] = action
        phase_usage[ts].append(action)

    # Collect waiting and queue stats
    total_wait = 0
    total_halt = 0
    lane_count = 0
    for lane in traci.lane.getIDList():
        total_wait += traci.lane.getWaitingTime(lane)
        total_halt += traci.lane.getLastStepHaltingNumber(lane)
        lane_count += 1
    avg_waiting_times.append(total_wait / lane_count)
    avg_queue_lengths.append(total_halt / lane_count)

    # Take step
    observations, rewards, done, _ = env.step(actions)
    rewards_per_step.append(sum(rewards.values()))
    step += 1

env.close()

# === Plotting Metrics ===
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(rewards_per_step)
plt.title("Total Reward Over Time")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(avg_waiting_times, label='Avg Waiting Time')
plt.plot(avg_queue_lengths, label='Avg Queue Length')
plt.title("Waiting Time and Queue Length")
plt.xlabel("Step")
plt.ylabel("Time / Queue")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
for ts in ts_ids:
    plt.hist(phase_usage[ts], bins=range(max(phase_usage[ts])+2), alpha=0.5, label=f"{ts}")
plt.title("Phase Usage Histogram")
plt.xlabel("Phase Index")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
