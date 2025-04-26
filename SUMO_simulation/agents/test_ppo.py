# ✅ test_ppo.py with per-road metric display and saved visualization
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sumo_rl
import traci
from train_ppo_agent import PPO, pad_state

print("✅ Starting PPO test...")

NETWORKS = [
    ("../nets/road1.net.xml", "../nets/road1.rou.xml"),
    ("../nets/road2.net.xml", "../nets/road2.rou.xml"),
    ("../nets/road3.net.xml", "../nets/road3.rou.xml")
]

MODEL_PATH = "../trained_models/ppo_model.pth"
MAX_STATE_DIM = 37
MAX_ACTION_DIM = 4
RESULT_DIR = "../experiment_results/ppo_agent"

os.makedirs(RESULT_DIR, exist_ok=True)


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


def test_ppo():
    print("📦 Loading PPO model...")
    agent = PPO(state_dim=MAX_STATE_DIM, max_action_dim=MAX_ACTION_DIM,
                hidden_size=64, lr=3e-4, gamma=0.99, clip_ratio=0.2, K_epoch=10)
    agent.actor_critic.load_state_dict(torch.load(MODEL_PATH))
    agent.actor_critic.eval()

    for net_file, route_file in NETWORKS:
        road_name = os.path.basename(net_file).replace(".net.xml", "")
        print(f"\n🚦 Evaluating PPO Agent on {road_name}")

        env = sumo_rl.SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            use_gui=False,
            num_seconds=3000,
            single_agent=False
        )

        obs = env.reset()
        done = {"__all__": False}
        traffic_signals = list(env.traffic_signals.keys())

        step = 0
        total_wait_time = 0
        total_queue_length = 0
        passed_vehicles = set()
        emergency_counts = []
        emergency_wait_logs = []

        while not done["__all__"]:
            actions = {}
            for ts in traffic_signals:
                padded_state = pad_state(obs[ts], MAX_STATE_DIM)
                action, _ = agent.choose_action(padded_state, env.action_space.n)
                actions[ts] = action

            obs, rewards, done, _ = env.step(actions)

            if traci.isLoaded():
                for lane_id in traci.lane.getIDList():
                    total_wait_time += traci.lane.getWaitingTime(lane_id)
                    total_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)

                for veh_id in traci.vehicle.getIDList():
                    if veh_id not in passed_vehicles and traci.vehicle.getRouteIndex(veh_id) > 0:
                        passed_vehicles.add(veh_id)

                emergency_counts.append(count_emergency_vehicles())
                emergency_wait_logs.extend([(step, veh_id, wait) for veh_id, wait in log_emergency_wait()])

            step += 1

        avg_wait = total_wait_time / step if step else 0
        avg_queue = total_queue_length / step if step else 0
        throughput = len(passed_vehicles)

        print(f"   ➤ Avg Waiting Time: {avg_wait:.2f} s")
        print(f"   ➤ Avg Queue Length: {avg_queue:.2f}")
        print(f"   ➤ Vehicles Passed:  {throughput}")
        print(f"   ➤ Avg Emergency Vehicles per Step: {np.mean(emergency_counts):.2f}")

        road_result_dir = os.path.join(RESULT_DIR, road_name)
        os.makedirs(road_result_dir, exist_ok=True)

        # Save time series plot
        plt.figure(figsize=(12, 8))
        plt.plot(emergency_counts, label='Emergency Vehicle Count', color='red')
        plt.title(f"PPO Agent Emergency Vehicle Detection: {road_name}")
        plt.xlabel("Step")
        plt.ylabel("Count")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(road_result_dir, "ppo_emergency_detection.png"))

        pd.DataFrame(emergency_wait_logs, columns=["step", "vehicle_id", "wait_time"]).to_csv(
            os.path.join(road_result_dir, "emergency_wait_log.csv"), index=False)

        with open(os.path.join(road_result_dir, "summary_metrics.txt"), "w", encoding="utf-8") as f:
            f.write(f"Summary for {road_name} (PPO)\n")
            f.write(f"Average Waiting Time: {avg_wait:.2f} s\n")
            f.write(f"Average Queue Length: {avg_queue:.2f}\n")
            f.write(f"Throughput: {throughput}\n")
            f.write(f"Average Emergency Vehicles per Step: {np.mean(emergency_counts):.2f}\n")

        env.close()


if __name__ == "__main__":
    test_ppo()
