# ✅ test_ppo.py with per-road metric display and saved visualization
import os
import torch
import numpy as np
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

def test_ppo():
    print("📦 Loading PPO model...")
    agent = PPO(state_dim=MAX_STATE_DIM, max_action_dim=MAX_ACTION_DIM,
                hidden_size=64, lr=3e-4, gamma=0.99, clip_ratio=0.2, K_epoch=10)
    agent.actor_critic.load_state_dict(torch.load(MODEL_PATH))
    agent.actor_critic.eval()

    all_waiting_times = []
    all_queue_lengths = []
    all_throughputs = []
    labels = []

    for net_file, route_file in NETWORKS:
        road_name = os.path.basename(net_file).replace(".net.xml", "")
        labels.append(road_name)
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

            step += 1

        avg_wait = total_wait_time / step if step else 0
        avg_queue = total_queue_length / step if step else 0
        throughput = len(passed_vehicles)

        print(f"   ➤ Avg Waiting Time: {avg_wait:.2f} s")
        print(f"   ➤ Avg Queue Length: {avg_queue:.2f}")
        print(f"   ➤ Vehicles Passed:  {throughput}")

        all_waiting_times.append(avg_wait)
        all_queue_lengths.append(avg_queue)
        all_throughputs.append(throughput)

        env.close()

    # Save bar chart
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, all_waiting_times, width=width, label='Avg Waiting Time')
    plt.bar(x, all_queue_lengths, width=width, label='Avg Queue Length')
    plt.bar(x + width, all_throughputs, width=width, label='Throughput')
    plt.xticks(x, labels)
    plt.xlabel("Road Network")
    plt.ylabel("Metric Value")
    plt.title("PPO Agent Performance per Road")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "ppo_performance_bar_chart.png"))
    print(f"📊 Saved performance chart to {RESULT_DIR}/ppo_performance_bar_chart.png")

    result = {
        "waiting_time": np.mean(all_waiting_times),
        "queue_length": np.mean(all_queue_lengths),
        "throughput": int(np.mean(all_throughputs))
    }

    print("\n📊 Overall PPO Performance:")
    print(result)
    return result

if __name__ == "__main__":
    test_ppo()
