import os
import numpy as np
import torch
from matplotlib import pylab as plt
import random
import sumo_rl
import traci
from ql_agent import QlAgent
import logging
from collections import defaultdict
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
)

def load_networks(directory: str) -> List[Dict[str, str]]:
    """
    Dynamically load network and route files from a directory.
    """
    networks = []
    for file in os.listdir(directory):
        if file.endswith(".net.xml"):
            net_file = os.path.join(directory, file)
            route_file = net_file.replace(".net.xml", ".rou.xml")
            if os.path.exists(route_file):
                networks.append({'net': net_file, 'route': route_file})
    return networks

def load_expected_signals(networks: List[Dict[str, str]]) -> Dict[str, List[str]]:
    """
    Generate expected signals for each network based on filenames.
    """
    expected_signals = {}
    for network in networks:
        network_name = os.path.splitext(os.path.basename(network['net']))[0]
        if network_name == "road2":
            # Define the actual traffic signal IDs for road2
            expected_signals[network_name] = ["TL"]  # road2 uses 'TL' as traffic light ID
        else:
            # Default assumption for other networks
            expected_signals[network_name] = [f"J{i}" for i in range(1, 8)]
    return expected_signals

def create_environment(net_file: str, route_file: str, seed: int) -> Any:
    """
    Create and return a new SumoEnvironment with common parameters.
    """
    env = sumo_rl.SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        num_seconds=3000,
        single_agent=False,
        add_system_info=True,
        add_per_agent_info=True,
        sumo_seed=seed,
        max_depart_delay=0,
        waiting_time_memory=100,
        delta_time=5
    )
    return env

def check_emergency_vehicle(env: Any, ts_id: str) -> int:
    """
    Returns the count of emergency vehicles detected at the given traffic signal.
    """
    emergency_count = 0
    try:
        if hasattr(env, 'get_emergency_vehicles'):
            emergency_vehicles = env.get_emergency_vehicles(ts_id)
            emergency_count = len(emergency_vehicles) if emergency_vehicles else 0
            if emergency_count > 0:
                logging.info(f"Emergency vehicles detected at {ts_id}: {emergency_count}")
    except Exception as e:
        logging.error(f"Error checking emergency vehicle at {ts_id}: {e}")
    return emergency_count

def pad_state(state: np.ndarray, max_state_dim: int) -> np.ndarray:
    """
    Pads smaller states with zeros to match max_state_dim.
    """
    padded_state = np.zeros(max_state_dim, dtype=np.float32)
    padded_state[:len(state)] = state
    return padded_state

def normalize_reward(reward: float, emergency_count: int = 0) -> float:
    """
    Normalize reward using a dynamic scaling approach with emergency vehicle consideration.
    """
    if np.isnan(reward):
        logging.warning("Encountered NaN reward. Setting reward to 0.")
        reward = 0.0
    scaled_reward = reward * 2 if reward <= 0 else reward
    if emergency_count > 0:
        scaled_reward *= (1 + emergency_count * 0.5)
    return scaled_reward

def evaluate_model(agent: QlAgent, env: Any, ts_id: str, max_state_dim: int) -> float:
    """
    Evaluate a model by running it in the environment and calculating the total reward.
    """
    total_reward = 0.0
    observations = env.reset()
    done = {"__all__": False}

    if ts_id not in observations:
        logging.error(f"Missing observations for {ts_id}.")
        return total_reward

    while not done["__all__"]:
        state = pad_state(observations[ts_id], max_state_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = torch.argmax(agent.predict_rewards(state_tensor)).item()
        next_observations, rewards, done, _ = env.step({ts_id: action})
        total_reward += rewards.get(ts_id, 0)
        observations = next_observations

    return total_reward

def compare_models(current_agent: QlAgent, best_model_path: str, env: Any, ts_id: str, max_state_dim: int) -> None:
    """
    Compare the current model with the best model saved so far.
    """
    if not os.path.exists(best_model_path):
        logging.info(f"No best model found for {ts_id}. Skipping comparison.")
        return

    best_model_state = torch.load(best_model_path)
    best_agent = QlAgent(input_shape=max_state_dim, output_shape=env.action_space.n)
    
    saved_input_shape = best_model_state['model_state_dict']['0.weight'].shape[1]
    if saved_input_shape != max_state_dim:
        logging.warning(f"Adjusting input shape for {ts_id}: Saved shape = {saved_input_shape}, Current shape = {max_state_dim}")
        best_agent.adjust_model(input_shape=saved_input_shape, output_shape=env.action_space.n)

    best_agent.model.load_state_dict(best_model_state['model_state_dict'])
    current_performance = evaluate_model(current_agent, env, ts_id, max_state_dim)
    best_performance = evaluate_model(best_agent, env, ts_id, saved_input_shape)
    logging.info(f"Comparison for {ts_id}: Current Model Reward = {current_performance}, Best Model Reward = {best_performance}")

def train() -> None:
    # Q-learning parameters
    epochs_per_network = 10
    epsilon = 1.0
    learning_rate = 1e-3
    min_epsilon = 0.01
    epsilon_decay = 0.995
    max_steps = 3000
    batch_size = 64
    memory_size = 10000

    os.makedirs('../trained_models', exist_ok=True)
    global_metrics = defaultdict(list)

    network_directory = '../nets'
    networks = load_networks(network_directory)
    expected_signals = load_expected_signals(networks)

    for network in networks:
        network_name = os.path.splitext(os.path.basename(network['net']))[0]
        logging.info(f"\n{'='*50}\nTraining on {network_name}\n{'='*50}")

        # Initialize environment to get dimensions
        env = create_environment(network['net'], network['route'], seed=42)
        max_state_dim = env.observation_space.shape[0]
        max_action_dim = env.action_space.n
        env.close()

        # Initialize agents for expected signals
        agents: Dict[str, QlAgent] = {}
        for ts in expected_signals[network_name]:
            agent = QlAgent(
                input_shape=max_state_dim,
                output_shape=max_action_dim,
                learning_rate=learning_rate,
                epsilon=epsilon,
                batch_size=batch_size,
                memory_size=memory_size
            )
            agents[ts] = agent
            logging.info(f"Initialized new agent for {ts}")

        reward_history: Dict[str, List[float]] = {ts: [] for ts in expected_signals[network_name]}
        best_rewards: Dict[str, float] = {ts: float('-inf') for ts in expected_signals[network_name]}
        current_epsilon = epsilon

        for epoch in range(epochs_per_network):
            logging.info(f"\nEpoch {epoch + 1}/{epochs_per_network} | Current epsilon: {current_epsilon:.3f}")
            env = create_environment(network['net'], network['route'], seed=42 + epoch)
            try:
                observations = env.reset()
                logging.info(f"Observations keys: {list(observations.keys())}")  # Debugging line
                available_signals = [ts for ts in expected_signals[network_name] if ts in observations]
                if not available_signals:
                    logging.error(f"No available signals found in environment for network {network_name}. Skipping training on this network.")
                    continue

                logging.info(f"Available signals for training: {available_signals}")
                done = {"__all__": False}
                step_count = 0
                episode_rewards: Dict[str, List[float]] = {ts: [] for ts in available_signals}

                while not done["__all__"] and step_count < max_steps:
                    step_count += 1
                    actions = {}
                    for ts_id in available_signals:
                        state = pad_state(observations[ts_id], max_state_dim)
                        emergency_count = check_emergency_vehicle(env, ts_id)
                        state_tensor = torch.tensor(state, dtype=torch.float32)

                        if random.random() < current_epsilon:
                            action = random.randint(0, max_action_dim - 1)
                        else:
                            with torch.no_grad():
                                q_values = agents[ts_id].predict_rewards(state_tensor)
                                action = torch.argmax(q_values).item()

                        if emergency_count > 0:
                            action = env.traffic_signals[ts_id].green_phase
                            logging.info(f"🚨 Emergency at {ts_id}, using phase {action}")

                        actions[ts_id] = action

                    next_observations, rewards, done, info = env.step(actions)

                    for ts_id in available_signals:
                        if ts_id not in next_observations or ts_id not in rewards:
                            logging.warning(f"Missing reward or observation for {ts_id}. Skipping this update.")
                            continue

                        state = pad_state(observations[ts_id], max_state_dim)
                        next_state = pad_state(next_observations[ts_id], max_state_dim)
                        emergency_count = check_emergency_vehicle(env, ts_id)
                        reward = normalize_reward(rewards[ts_id], emergency_count)
                        agents[ts_id].remember(state, actions[ts_id], reward, next_state, done["__all__"])
                        if len(agents[ts_id].memory) >= batch_size:
                            with torch.no_grad():
                                state_tensor = torch.tensor(state, dtype=torch.float32)
                                pred_reward = agents[ts_id].predict_rewards(state_tensor)
                            agents[ts_id].learn(pred_reward, reward, emergency_count > 0)
                        episode_rewards[ts_id].append(reward)

                    observations = next_observations

                    if step_count % 100 == 0:
                        logging.info(f"\nStep {step_count}/{max_steps}")
                        for ts_id in episode_rewards:
                            avg_reward = np.mean(episode_rewards[ts_id][-100:])
                            logging.info(f"{ts_id} - Avg Reward (last 100): {avg_reward:.2f}")
            finally:
                env.close()

            for ts_id in available_signals:
                avg_reward = np.mean(episode_rewards[ts_id])
                reward_history[ts_id].append(avg_reward)
                global_metrics[f"{network_name}_{ts_id}"].append(avg_reward)
                model_path = f"../trained_models/ql_model_final_{network_name}_{ts_id}.pth"
                if avg_reward > best_rewards[ts_id]:
                    best_rewards[ts_id] = avg_reward
                    torch.save({
                        'model_state_dict': agents[ts_id].model.state_dict(),
                        'reward_mean': agents[ts_id].reward_mean,
                        'reward_std': agents[ts_id].reward_std,
                        'reward_count': agents[ts_id].reward_count
                    }, model_path)
                    logging.info(f"Saved best model for {ts_id} (reward: {avg_reward:.2f})")
                compare_models(agents[ts_id], model_path, env, ts_id, max_state_dim)

            current_epsilon = max(min_epsilon, current_epsilon * epsilon_decay)

        # Plot training progress for this network
        plt.figure(figsize=(12, 6))
        for ts_id in reward_history:
            plt.plot(reward_history[ts_id], label=ts_id)
        plt.title(f'Training Progress on {network_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.legend()
        plt.savefig(f'../trained_models/training_progress_{network_name}.png')
        plt.close()

    # Plot global training progress
    plt.figure(figsize=(15, 8))
    for metric_name, values in global_metrics.items():
        plt.plot(values, label=metric_name)
    plt.title('Global Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Average Reward')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('../trained_models/global_training_progress.png')
    plt.close()

if __name__ == "__main__":
    train()
