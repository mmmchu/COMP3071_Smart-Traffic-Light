import os
import sys
import torch
import logging
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sumo_rl
from pathlib import Path

# Ensure the experiment_results directory exists
experiment_results_dir = Path('../experiment_results')
experiment_results_dir.mkdir(parents=True, exist_ok=True)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(experiment_results_dir / 'experiment.log'),
        logging.StreamHandler()
    ]
)

class TrafficMetrics:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics = {
            'waiting_time': [],
            'queue_length': [],
            'vehicle_count': [],
            'emergency_count': [],
            'avg_speed': [],
            'total_vehicles': []
        }
        self.current_episode = 0
        self.step = 0
        
    def update(self, env):
        """Update metrics with current simulation state"""
        try:
            import traci
            if not traci.isLoaded():
                logging.error("TraCI not loaded in update method")
                return
                
            # Get all vehicles in the simulation
            vehicles = traci.vehicle.getIDList()
            if not vehicles:
                logging.warning("No vehicles detected in simulation")
                return
                
            # Calculate metrics
            total_waiting = 0
            total_queue = 0
            total_speed = 0
            emergency_count = 0
            
            for vehicle in vehicles:
                try:
                    # Get vehicle information
                    waiting_time = traci.vehicle.getWaitingTime(vehicle)
                    speed = traci.vehicle.getSpeed(vehicle)
                    lane = traci.vehicle.getLaneID(vehicle)
                    
                    # All lanes are controlled in this case
                    total_waiting += waiting_time
                    if speed < 0.1:  # Vehicle is considered queued if speed is very low
                        total_queue += 1
                    total_speed += speed
                    
                    # Check if vehicle is emergency vehicle
                    if traci.vehicle.getTypeID(vehicle) == 'emergency':
                        emergency_count += 1
                            
                except Exception as e:
                    logging.warning(f"Error processing vehicle {vehicle}: {e}")
                    continue
            
            # Update metrics
            self.metrics['waiting_time'].append(total_waiting)
            self.metrics['queue_length'].append(total_queue)
            self.metrics['vehicle_count'].append(len(vehicles))
            self.metrics['emergency_count'].append(emergency_count)
            self.metrics['avg_speed'].append(total_speed / len(vehicles) if vehicles else 0)
            self.metrics['total_vehicles'].append(len(vehicles))
            
            # Log metrics every 100 steps
            if self.step % 100 == 0:
                logging.info(f"Step {self.step} metrics:")
                logging.info(f"Vehicles: {len(vehicles)}")
                logging.info(f"Waiting time: {total_waiting:.2f}")
                logging.info(f"Queue length: {total_queue}")
                logging.info(f"Average speed: {total_speed / len(vehicles) if vehicles else 0:.2f}")
                logging.info(f"Emergency vehicles: {emergency_count}")
            
            self.step += 1
            
        except Exception as e:
            logging.error(f"Error updating metrics: {e}")

    def plot_metrics(self, episode, save_path=None):
        """Plot metrics for the current episode"""
        try:
            plt.figure(figsize=(15, 10))
            
            # Plot waiting times
            plt.subplot(2, 2, 1)
            plt.plot(self.metrics['waiting_time'])
            plt.title(f'Episode {episode} - Total Waiting Time')
            plt.xlabel('Step')
            plt.ylabel('Time (s)')
            
            # Plot queue lengths
            plt.subplot(2, 2, 2)
            plt.plot(self.metrics['queue_length'])
            plt.title(f'Episode {episode} - Queue Length')
            plt.xlabel('Step')
            plt.ylabel('Vehicles')
            
            # Plot vehicle counts
            plt.subplot(2, 2, 3)
            plt.plot(self.metrics['vehicle_count'])
            plt.title(f'Episode {episode} - Vehicle Count')
            plt.xlabel('Step')
            plt.ylabel('Vehicles')
            
            # Plot emergency vehicle counts
            plt.subplot(2, 2, 4)
            plt.plot(self.metrics['emergency_count'])
            plt.title(f'Episode {episode} - Emergency Vehicles')
            plt.xlabel('Step')
            plt.ylabel('Vehicles')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.close()
            
            # Save metrics to CSV
            self.save_metrics(episode)
            
        except Exception as e:
            logging.error(f"Error plotting metrics: {str(e)}")
    
    def save_metrics(self, episode):
        """Save metrics to CSV file"""
        try:
            import pandas as pd
            
            # Convert metrics to DataFrame
            df = pd.DataFrame(self.metrics)
            
            # Save to CSV
            csv_path = os.path.join(self.output_dir, f'metrics_episode_{episode}.csv')
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved metrics to {csv_path}")
            
        except Exception as e:
            logging.error(f"Error saving metrics: {str(e)}")
    
    def get_metrics(self):
        """Get current metrics"""
        return {k: v[-1] if v else 0 for k, v in self.metrics.items()}

def load_trained_agent(model_path, input_shape, output_shape):
    """Load a trained agent from a checkpoint file."""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path)
        
        # Add the parent directory to Python path to find the agents module
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from agents.ql_agent import QlAgent
        
        agent = QlAgent(input_shape=input_shape, output_shape=output_shape)
        
        ## Del two lines below to avoid loading the model weights
        # Log input shape for debugging
        logging.info(f"Model input shape: {input_shape}, Checkpoint input shape: {checkpoint['model_state_dict']['0.weight'].shape[1]}")
        
        # Check for input size mismatch
        if checkpoint['model_state_dict']['0.weight'].shape[1] != input_shape:
            raise ValueError(
                f"Input size mismatch: Model expects {input_shape}, "
                f"but checkpoint has {checkpoint['model_state_dict']['0.weight'].shape[1]}"
            )
        
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.model.eval()
        
        logging.info(f"Successfully loaded model from {model_path}")
        return agent
        
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise

def run_experiment(
    net_file: str,
    route_file: str,
    model_path: str,
    num_episodes: int = 5,
    num_seconds: int = 20000,  # Increased simulation time
    yellow_time: int = 2,
    min_green: int = 10,
    max_green: int = 50,
    single_agent: bool = False,
    output_dir: str = "../experiment_results"
):
    """
    Run an experiment using a trained agent.
    
    Args:
        net_file: Path to the SUMO network file
        route_file: Path to the SUMO route file
        model_path: Path to the trained model
        num_episodes: Number of episodes to run
        num_seconds: Number of seconds to run each episode
        yellow_time: Duration of yellow phase
        min_green: Minimum duration of green phase
        max_green: Maximum duration of green phase
        single_agent: Whether to use a single agent for all junctions
        output_dir: Directory to save experiment results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'experiment.log')),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting experiment with:")
    logging.info(f"Network file: {net_file}")
    logging.info(f"Route file: {route_file}")
    logging.info(f"Model path: {model_path}")
    
    # Verify files exist
    for file_path in [net_file, route_file, model_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Log route file contents
    with open(route_file, 'r') as f:
        logging.info("Route file contents:")
        logging.info(f.read())
    
    # Initialize metrics collector
    metrics_collector = TrafficMetrics(output_dir=output_dir)
    
    # Create SUMO environment
    logging.info("Creating SUMO environment...")
    try:
        env = sumo_rl.SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            num_seconds=num_seconds,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            single_agent=single_agent,
            use_gui=False,
            additional_sumo_cmd="--no-step-log --no-warnings --log " + os.path.join(output_dir, "sumo.log")
        )
        logging.info("SUMO environment created successfully")
    except Exception as e:
        logging.error(f"Failed to create SUMO environment: {str(e)}")
        raise
    
    # Load trained agent
    agent = load_trained_agent(model_path, env.observation_space.shape[0], env.action_space.n)
    
    # Run experiment
    for episode in range(num_episodes):
        logging.info(f"Starting episode {episode + 1}/{num_episodes}")
        state = env.reset()
        done = False
        step = 0
        
        while not done:
            # Get action from agent and convert to proper format
            action_tensor = agent.predict_rewards(torch.tensor(state['J1'], dtype=torch.float32))
            action_idx = torch.argmax(action_tensor).item()
            action = {'J1': action_idx}  # Convert to dictionary format
            
            next_state, reward, done, info = env.step(action)
            
            # Log action and reward
            logging.info(f"Step {step}: Action={action}, Reward={reward}")
            
            # Update metrics
            metrics_collector.update(env)
            
            state = next_state
            step += 1
            
            if step % 100 == 0:
                logging.info(f"Episode {episode + 1}, Step {step}")
                logging.info(f"Metrics: {metrics_collector.get_metrics()}")
        
        # Save metrics for this episode
        metrics_collector.plot_metrics(episode + 1, os.path.join(output_dir, f'metrics_episode_{episode+1}.png'))
    
    env.close()
    logging.info("Experiment completed")

if __name__ == "__main__":
    try:
        # Get the absolute path to the script directory
        script_dir = Path(__file__).resolve().parent
        
        # Define experiment configuration
        road = 'road1'        # Example: 'road1', 'road2', or 'road3'
        junction = 'J1'       # For road3, valid options might be: 'J1', 'J4', 'J5', 'J6', or 'J7'
        
        # Create a unique output directory for this experiment
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{road}_{junction}_{timestamp}"
        output_dir = script_dir.parent / 'experiment_results' / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up paths for the network, route, and model files dynamically
        net_file = script_dir.parent / 'nets' / f'{road}.net.xml'
        route_file = script_dir.parent / 'nets' / f'{road}.rou.xml'
        
        # Use ql_model_final_road3.net_J1.pth for Road2 testing
        if road == 'road2':
            model_path = script_dir.parent / 'trained_models' / 'ql_model_final_road3.net_J1.pth'
        else:
            model_path = script_dir.parent / 'trained_models' / f'ql_model_final_{road}.net_{junction}.pth'
        
        # Validate paths
        for file_path in [net_file, route_file, model_path]:
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Log paths for verification
        logging.info(f"Network file: {net_file}")
        logging.info(f"Route file: {route_file}")
        logging.info(f"Model file: {model_path}")
        logging.info(f"Output directory: {output_dir}")
        
        # Run the experiment using the generated output directory
        run_experiment(
            net_file=str(net_file),
            route_file=str(route_file),
            model_path=str(model_path),
            num_episodes=5,
            num_seconds=10000,
            yellow_time=2,
            min_green=10,
            max_green=50,
            single_agent=False,
            output_dir=str(output_dir)
        )
        
    except Exception as e:
        logging.error(f"Failed to run experiment: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)