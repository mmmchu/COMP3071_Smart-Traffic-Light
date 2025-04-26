import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sumo_rl
import traci

from ql_agent import QlAgent
from train_ppo_agent import PPO, pad_state

# Test Configuration
NUM_EPISODES = 10  # Number of episodes to run for each agent on each network
SIMULATION_TIME = 3000  # Duration of each episode in seconds

NETWORKS = [
    ("../nets/road1.net.xml", "../nets/road1.rou.xml"),
    ("../nets/road2.net.xml", "../nets/road2.rou.xml"),
    ("../nets/road3.net.xml", "../nets/road3.rou.xml")
]

MODEL_PATHS = {
    "ppo": "../trained_models/ppo_model.pth",
    "ql": "../trained_models/ql_model.pth"
}

MAX_STATE_DIM = 37
MAX_Q_STATE_DIM = 49
MAX_ACTION_DIM = 4

RESULT_DIR = "../experiment_results/all_agents"
os.makedirs(RESULT_DIR, exist_ok=True)

def count_emergency_vehicles():
    count = 0
    for lane in traci.lane.getIDList():
        for v in traci.lane.getLastStepVehicleIDs(lane):
            if traci.vehicle.getTypeID(v) == "DEFAULT_CONTAINERTYPE":
                count += 1
    return count

def run_episode(agent_type, net_file, route_file):
    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=SIMULATION_TIME,
        single_agent=False
    )

    obs = env.reset()
    done = {"__all__": False}
    traffic_signals = list(env.traffic_signals.keys())

    if agent_type == "ppo":
        agent = PPO(state_dim=MAX_STATE_DIM, max_action_dim=MAX_ACTION_DIM,
                    hidden_size=64, lr=3e-4, gamma=0.99, clip_ratio=0.2, K_epoch=10)
        agent.actor_critic.load_state_dict(torch.load(MODEL_PATHS["ppo"]))
        agent.actor_critic.eval()
    elif agent_type == "ql":
        agent = QlAgent(input_shape=MAX_Q_STATE_DIM)
        agent.model.load_state_dict(torch.load(MODEL_PATHS["ql"]))
        agent.model.eval()

    step = 0
    total_wait_time = 0
    total_queue_length = 0
    passed_vehicles = set()
    emergency_counts = []

    while not done["__all__"]:
        actions = {}
        for ts in traffic_signals:
            if agent_type == "ppo":
                padded_state = pad_state(obs[ts], MAX_STATE_DIM)
                action, _ = agent.choose_action(padded_state, env.action_space.n)
            elif agent_type == "ql":
                emergency = count_emergency_vehicles()
                padded_state = np.pad(np.append(obs[ts], emergency), (0, MAX_Q_STATE_DIM - len(obs[ts]) - 1))
                state_tensor = torch.tensor(padded_state, dtype=torch.float32)

                ts_obj = env.traffic_signals[ts]
                current_phase = ts_obj.green_phase
                valid_transitions = [new for (old, new) in ts_obj.yellow_dict.keys() if old == current_phase]

                pred = agent.predict_rewards(state_tensor)
                masked_pred = torch.full_like(pred, float('-inf'))
                for phase in valid_transitions:
                    masked_pred[phase] = pred[phase]

                action = torch.argmax(masked_pred).item()
            else:
                action = None

            if action is not None:
                actions[ts] = action

        obs, rewards, done, _ = env.step(actions if actions else {})

        if traci.isLoaded():
            for lane_id in traci.lane.getIDList():
                total_wait_time += traci.lane.getWaitingTime(lane_id)
                total_queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
            for veh_id in traci.vehicle.getIDList():
                if traci.vehicle.getRouteIndex(veh_id) > 0:
                    passed_vehicles.add(veh_id)
            emergency_counts.append(count_emergency_vehicles())

        step += 1

    env.close()

    avg_wait = total_wait_time / step if step else 0
    avg_queue = total_queue_length / step if step else 0
    throughput = len(passed_vehicles)
    avg_emergency = np.mean(emergency_counts)

    return avg_wait, avg_queue, throughput, avg_emergency

def run_agent(agent_type, road_name, net_file, route_file):
    print(f"\n🚦 Testing {agent_type.upper()} agent on {road_name}...")
    
    # Run multiple episodes and collect results
    episode_results = {
        "wait": [],
        "queue": [],
        "throughput": [],
        "emergency": []
    }
    
    for episode in range(NUM_EPISODES):
        print(f"   Episode {episode + 1}/{NUM_EPISODES}")
        avg_wait, avg_queue, throughput, avg_emergency = run_episode(agent_type, net_file, route_file)
        
        episode_results["wait"].append(avg_wait)
        episode_results["queue"].append(avg_queue)
        episode_results["throughput"].append(throughput)
        episode_results["emergency"].append(avg_emergency)
        
        print(f"   ➤ Avg Wait Time: {avg_wait:.2f} s | Queue: {avg_queue:.2f} | Throughput: {throughput} | Emergencies/Step: {avg_emergency:.2f}")
    
    # Calculate mean and standard deviation
    mean_wait = np.mean(episode_results["wait"])
    std_wait = np.std(episode_results["wait"])
    mean_queue = np.mean(episode_results["queue"])
    std_queue = np.std(episode_results["queue"])
    mean_throughput = np.mean(episode_results["throughput"])
    std_throughput = np.std(episode_results["throughput"])
    mean_emergency = np.mean(episode_results["emergency"])
    std_emergency = np.std(episode_results["emergency"])
    
    print(f"\n   📊 Summary over {NUM_EPISODES} episodes:")
    print(f"   ➤ Avg Wait Time: {mean_wait:.2f} ± {std_wait:.2f} s")
    print(f"   ➤ Avg Queue Length: {mean_queue:.2f} ± {std_queue:.2f}")
    print(f"   ➤ Avg Throughput: {mean_throughput:.2f} ± {std_throughput:.2f}")
    print(f"   ➤ Avg Emergencies/Step: {mean_emergency:.2f} ± {std_emergency:.2f}")
    
    return mean_wait, mean_queue, mean_throughput, mean_emergency, std_wait, std_queue, std_throughput, std_emergency

def test_all():
    summary_rows = []
    agents = ["fixed", "ql", "ppo"]
    metrics = {a: {"wait": [], "queue": [], "throughput": [], "emergency": [],
                  "wait_std": [], "queue_std": [], "throughput_std": [], "emergency_std": []} 
              for a in agents}

    for net_file, route_file in NETWORKS:
        road_name = os.path.basename(net_file).replace(".net.xml", "")
        for agent in agents:
            mean_wait, mean_queue, mean_throughput, mean_emergency, \
            std_wait, std_queue, std_throughput, std_emergency = run_agent(agent, road_name, net_file, route_file)
            
            metrics[agent]["wait"].append(mean_wait)
            metrics[agent]["queue"].append(mean_queue)
            metrics[agent]["throughput"].append(mean_throughput)
            metrics[agent]["emergency"].append(mean_emergency)
            metrics[agent]["wait_std"].append(std_wait)
            metrics[agent]["queue_std"].append(std_queue)
            metrics[agent]["throughput_std"].append(std_throughput)
            metrics[agent]["emergency_std"].append(std_emergency)
            
            summary_rows.append({
                "Road": road_name,
                "Agent": agent,
                "Avg Wait Time (s)": mean_wait,
                "Wait Time Std": std_wait,
                "Avg Queue Length": mean_queue,
                "Queue Length Std": std_queue,
                "Throughput": mean_throughput,
                "Throughput Std": std_throughput,
                "Emergencies per Step": mean_emergency,
                "Emergencies Std": std_emergency
            })

    labels = [os.path.basename(net_file).replace(".net.xml", "") for net_file, _ in NETWORKS]
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(14, 6))
    for i, metric in enumerate(["wait", "queue", "throughput", "emergency"]):
        plt.subplot(1, 4, i+1)
        for j, agent in enumerate(agents):
            means = metrics[agent][metric]
            stds = metrics[agent][f"{metric}_std"]
            plt.bar(x + j*width - width, means, width, label=agent, yerr=stds, capsize=5)
        plt.xticks(x, labels)
        plt.title(metric.capitalize())
        plt.xlabel("Road")
        plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "all_agents_comparison.png"))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(RESULT_DIR, "summary_results.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print("\n📄 Summary CSV saved to", summary_csv_path)

if __name__ == "__main__":
    test_all()