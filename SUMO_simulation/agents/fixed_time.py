import os

# ✅ Set this BEFORE importing sumo_rl to inject additional files
os.environ["SUMO_ADDITIONAL_FILES"] = "../nets/fixed_tls.add.xml"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traci
from sumo_rl import SumoEnvironment

networks = [
    ("../nets/road1.net.xml", "../nets/road1.rou.xml", "road1"),
    ("../nets/road2.net.xml", "../nets/road2.rou.xml", "road2"),
    ("../nets/road3.net.xml", "../nets/road3.rou.xml", "road3"),
]

result_dir = "../experiment_results/fixed_agent"
os.makedirs(result_dir, exist_ok=True)

def count_emergency_vehicles():
    count = 0
    for lane in traci.lane.getIDList():
        for v in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getTypeID(v) == "DEFAULT_CONTAINERTYPE":
                count += 1
    return count

def log_emergency_wait():
    logs = []
    for lane in traci.lane.getIDList():
        for veh_id in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getTypeID(veh_id) == "DEFAULT_CONTAINERTYPE":
                wait = traci.vehicle.getWaitingTime(veh_id)
                logs.append((veh_id, wait))
    return logs

for net_file, route_file, road_id in networks:
    print(f"\n🚦 Evaluating Fixed-Time Agent on {road_id}")

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=True,
        num_seconds=3000,
        single_agent=False
    )

    obs = env.reset()
    ts_ids = list(env.traffic_signals.keys())
    print("Traffic signals:", ts_ids)

    done = {"__all__": False}
    step = 0
    rewards = []
    avg_waiting_times = []
    avg_queue_lengths = []
    emergency_counts = []
    emergency_wait_logs = []

    while not done["__all__"]:
        obs, reward, done, _ = env.step({})  # No actions = fixed schedule

        total_wait = sum(traci.lane.getWaitingTime(l) for l in traci.lane.getIDList())
        total_queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in traci.lane.getIDList())
        lane_count = len(traci.lane.getIDList())

        avg_waiting_times.append(total_wait / lane_count if lane_count else 0)
        avg_queue_lengths.append(total_queue / lane_count if lane_count else 0)
        rewards.append(sum(reward.values()))

        emergency_counts.append(count_emergency_vehicles())
        emergency_wait_logs.extend([(step, veh_id, wait) for veh_id, wait in log_emergency_wait()])

        step += 1

    env.close()

    # === Save plot and metrics
    plt.figure(figsize=(12, 8))
    plt.plot(rewards, label="Reward")
    plt.plot(avg_waiting_times, label="Avg Waiting Time")
    plt.plot(avg_queue_lengths, label="Avg Queue Length")
    plt.plot(emergency_counts, label="Emergency Vehicle Count", color="red")
    plt.title(f"Fixed-Time Signal Performance: {road_id}")
    plt.xlabel("Step")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    road_result_dir = os.path.join(result_dir, road_id)
    os.makedirs(road_result_dir, exist_ok=True)

    plt.savefig(os.path.join(road_result_dir, "fixed_stepwise_metrics.png"))
    print(f"📊 Plot saved to {road_result_dir}/fixed_stepwise_metrics.png")

    pd.DataFrame({
        "step": np.arange(len(rewards)),
        "reward": rewards,
        "waiting_time": avg_waiting_times,
        "queue_length": avg_queue_lengths,
        "emergency_count": emergency_counts
    }).to_csv(os.path.join(road_result_dir, "step_metrics.csv"), index=False)

    pd.DataFrame(emergency_wait_logs, columns=["step", "vehicle_id", "wait_time"]).to_csv(
        os.path.join(road_result_dir, "emergency_wait_log.csv"), index=False)

    summary_path = os.path.join(road_result_dir, "summary_metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Summary for {road_id} (Fixed-Time)\n")
        f.write(f"Average Reward: {np.mean(rewards):.2f}\n")
        f.write(f"Average Waiting Time: {np.mean(avg_waiting_times):.2f} s\n")
        f.write(f"Average Queue Length: {np.mean(avg_queue_lengths):.2f}\n")
        f.write(f"Average Emergency Vehicles per Step: {np.mean(emergency_counts):.2f}\n")
        print(f"📄 Summary saved to {summary_path}")
