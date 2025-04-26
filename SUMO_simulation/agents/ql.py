import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sumo_rl
import traci
from ql_agent import QlAgent

# === Test Configuration ===
NETWORKS = [
    ("../nets/road1.net.xml", "../nets/road1.rou.xml"),
    ("../nets/road2.net.xml", "../nets/road2.rou.xml"),
    ("../nets/road3.net.xml", "../nets/road3.rou.xml")
]

model_path = "../trained_models/ql_model.pth"
max_input_shape = 49
RESULT_DIR = "../experiment_results/ql_agent"
os.makedirs(RESULT_DIR, exist_ok=True)

def check_emergency(ts):
    count = 0
    for lane in traci.trafficlight.getControlledLanes(ts):
        for v in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getTypeID(v) == "DEFAULT_CONTAINERTYPE":
                count += 1
    return count

def test_ql():
    print("📦 Loading QL model...")
    agent = QlAgent(input_shape=max_input_shape)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()

    for net_file, route_file in NETWORKS:
        road_name = os.path.basename(net_file).replace(".net.xml", "")
        print(f"\n🚦 Evaluating QL Agent on {road_name}")

        # === Setup Environment ===
        env = sumo_rl.SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=True,
            num_seconds=5000,
            single_agent=False
        )

        observations = env.reset()
        ts_ids = list(env.traffic_signals.keys())
        print("Traffic signals:", ts_ids)

        # === Tracking Metrics ===
        rewards_per_step = []
        avg_waiting_times = []
        avg_queue_lengths = []
        emergency_vehicles = []
        emergency_wait_logs = []
        phase_usage = {ts: [] for ts in ts_ids}

        # === Simulation Loop ===
        done = {"__all__": False}
        step = 0
        while not done["__all__"]:
            actions = {}
            total_emergency = 0

            for ts in ts_ids:
                emergency = check_emergency(ts)
                total_emergency += emergency

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

            emergency_vehicles.append(total_emergency)

            # Emergency wait time tracking
            for lane in traci.lane.getIDList():
                for veh_id in traci.lane.getLastStepVehicleIDs(lane):
                    if traci.vehicle.getTypeID(veh_id) == "DEFAULT_CONTAINERTYPE":
                        wait_time = traci.vehicle.getWaitingTime(veh_id)
                        emergency_wait_logs.append((step, veh_id, wait_time))

            # Collect waiting and queue stats
            total_wait = 0
            total_halt = 0
            lane_count = 0
            for lane in traci.lane.getIDList():
                total_wait += traci.lane.getWaitingTime(lane)
                total_halt += traci.lane.getLastStepHaltingNumber(lane)
                lane_count += 1
            avg_waiting_times.append(total_wait / lane_count if lane_count else 0)
            avg_queue_lengths.append(total_halt / lane_count if lane_count else 0)

            # Take step
            observations, rewards, done, _ = env.step(actions)
            rewards_per_step.append(sum(rewards.values()))
            step += 1

        env.close()

        # === Save Results ===
        road_result_dir = os.path.join(RESULT_DIR, road_name)
        os.makedirs(road_result_dir, exist_ok=True)

        # === Plotting Metrics ===
        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(rewards_per_step)
        plt.title(f"Total Reward Over Time - {road_name}")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(avg_waiting_times, label='Avg Waiting Time')
        plt.plot(avg_queue_lengths, label='Avg Queue Length')
        plt.title("Waiting Time and Queue Length")
        plt.xlabel("Step")
        plt.ylabel("Time / Queue")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        for ts in ts_ids:
            plt.hist(phase_usage[ts], bins=range(max(phase_usage[ts]) + 2), alpha=0.5, label=f"{ts}")
        plt.title("Phase Usage Histogram")
        plt.xlabel("Phase Index")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
        plt.plot(emergency_vehicles, label='Emergency Vehicle Count', color='red')
        plt.title("Emergency Vehicles Detected")
        plt.xlabel("Step")
        plt.ylabel("Count")
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(road_result_dir, "ql_stepwise_metrics.png")
        plt.savefig(plot_path)
        print(f"\U0001F4CA Saved stepwise QL performance to {plot_path}")

        # === CSV Output ===
        df = pd.DataFrame({
            "step": np.arange(len(rewards_per_step)),
            "reward": rewards_per_step,
            "waiting_time": avg_waiting_times,
            "queue_length": avg_queue_lengths,
            "emergency_count": emergency_vehicles
        })
        csv_path = os.path.join(road_result_dir, "step_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\U0001F4C4 Saved stepwise QL metrics CSV to {csv_path}")

        # === Save emergency wait log ===
        emergency_log_path = os.path.join(road_result_dir, "emergency_wait_log.csv")
        pd.DataFrame(emergency_wait_logs, columns=["step", "vehicle_id", "wait_time"]).to_csv(emergency_log_path, index=False)
        print(f"\U0001F4C4 Saved emergency wait log to {emergency_log_path}")

        # === Summary Output ===
        summary_path = os.path.join(road_result_dir, "summary_metrics.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"\U0001F4CA Summary QL Metrics for {road_name}:\n")
            f.write(f"Average Reward per Step: {np.mean(rewards_per_step):.2f}\n")
            f.write(f"Average Waiting Time: {np.mean(avg_waiting_times):.2f} s\n")
            f.write(f"Average Queue Length: {np.mean(avg_queue_lengths):.2f}\n")
            f.write(f"Average Emergency Vehicles per Step: {np.mean(emergency_vehicles):.2f}\n")
        print(f"\U0001F4C4 Saved summary QL metrics to {summary_path}")

        # === Print Summary to Console ===
        print(f"\n\U0001F4CA Summary QL Metrics for {road_name}:")
        print(f"Average Reward per Step: {np.mean(rewards_per_step):.2f}")
        print(f"Average Waiting Time: {np.mean(avg_waiting_times):.2f} s")
        print(f"Average Queue Length: {np.mean(avg_queue_lengths):.2f}")
        print(f"Average Emergency Vehicles per Step: {np.mean(emergency_vehicles):.2f}")

if __name__ == "__main__":
    test_ql()
