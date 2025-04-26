# Smart Traffic Light Control System

This project implements a smart traffic light control system using reinforcement learning algorithms. The system aims to optimise traffic flow at intersections by dynamically adjusting traffic light timings based on real-time traffic conditions.

## Project Structure

The project consists of two main simulation environments:
- `pygame_simulation/`: A 2D visualisation of the traffic intersection using Pygame
- `SUMO_simulation/`: A more realistic traffic simulation using SUMO (Simulation of Urban MObility)

## Features

- Implementation of two reinforcement learning algorithms:
  - Q-Learning (QL)
  - Proximal Policy Optimization (PPO)
- Fixed-time traffic light control (baseline)
- Real-time traffic visualization
- Configurable simulation parameters
- Performance metrics tracking and visualization
- Emergency vehicle detection and handling
- Multiple vehicle types support (cars, buses, trucks, bikes)

## Requirements

- Python 3.x
- Pygame 2.6.1
- PyTorch 2.6.0
- NumPy 2.2.4
- Matplotlib 3.10.1
- SUMO (for SUMO simulation)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Pygame Simulation
To run the Pygame-based simulation:
```bash
cd pygame_simulation
python main.py
```

### SUMO Simulation
The SUMO simulation provides three different traffic light control methods:

1. Fixed-time Control:
```bash
cd SUMO_simulation
python fixed.py
```

2. Q-Learning Control:
```bash
cd SUMO_simulation
python ql.py
```

3. PPO Control:
```bash
cd SUMO_simulation
python ppo.py
```

### Testing Agent Performance
To test and compare the performance of all agents:
```bash
cd SUMO_simulation/agents
python test_agent.py
```

This will:
- Test three types of traffic light control:
  1. Fixed-time (baseline)
  2. Q-Learning
  3. PPO
- Run multiple episodes (10) for statistical significance
- Test on multiple road networks:
  1. road1.net.xml
  2. road2.net.xml
  3. road3.net.xml
- Generate performance comparisons
- Create visualizations and summary reports
- Save results in "experiment_result"

## File Descriptions

### Pygame Simulation Files
- `main.py`: Main simulation file that handles:
  - Traffic signal control
  - Vehicle generation and movement
  - Real-time visualization
  - Performance metrics collection
- `vehicle.py`: Contains vehicle class and related functions:
  - Vehicle movement logic
  - Collision detection
  - Traffic rules implementation
- `menuUI.py`: User interface for simulation configuration:
  - Traffic spawn rate selection
  - Simulation parameters adjustment
- `config.py`: Configuration settings for the simulation:
  - Traffic signal timings
  - Vehicle parameters
  - Simulation constants

### SUMO Simulation Files
- `fixed.py`: Fixed-time traffic light control:
  - Predefined signal timings
  - Baseline for comparison
  - Simple control strategy
- `ql.py`: Q-Learning implementation:
  - Q-table/neural network based control
  - Epsilon-greedy exploration
  - Real-time learning
- `ppo.py`: PPO implementation:
  - Actor-Critic architecture
  - Policy gradient optimization
  - Continuous action space
- `test_agent.py`: Comprehensive agent testing and evaluation:
  - Tests all three control methods in one run
  - Runs multiple episodes (10) for statistical significance
  - Tests on multiple road networks
  - Generates comparative performance metrics and visualizations
  - Saves results in a structured format

### Agent Implementation Files
- `ql_agent_pseudocode.txt`: Q-Learning agent implementation:
  - Q-table/neural network structure
  - Epsilon-greedy exploration
  - Q-value updates
- `ppo_agent_pseudocode.txt`: PPO agent implementation:
  - Policy and value networks
  - Clipped surrogate objective
  - Advantage estimation
- `train_ql_pseudocode.txt`: Q-Learning training process:
  - Experience collection
  - Q-value updates
  - Model saving
- `train_ppo_pseudocode.txt`: PPO training process:
  - Trajectory collection
  - Policy updates
  - Value function updates

## Training Process

### Q-Learning Training
The Q-Learning training process (`train_ql_pseudocode.txt`) follows these steps:

1. **Initialization**:
   - Initialize Q-table or neural network
   - Set hyperparameters:
     - Learning rate (α)
     - Discount factor (γ)
     - Exploration rate (ε)
     - Minimum exploration rate
     - Exploration decay rate

2. **Training Loop**:
   - For each episode:
     a. Reset environment and get initial state
     b. While episode not done:
        - Select action using ε-greedy policy
        - Execute action and observe reward, next state
        - Update Q-values using Bellman equation:
          ```
          Q(s,a) = Q(s,a) + α[r + γ max(Q(s',a')) - Q(s,a)]
          ```
        - Decay exploration rate ε
     c. Save model periodically
     d. Log performance metrics

3. **Performance Metrics**:
   - Average reward per episode
   - Queue lengths
   - Waiting times
   - Flow rates
   - Emergency vehicle response times

### PPO Training
The PPO training process (`train_ppo_pseudocode.txt`) follows these steps:

1. **Initialization**:
   - Initialize policy network (actor) and value network (critic)
   - Set hyperparameters:
     - Learning rate
     - Discount factor (γ)
     - GAE lambda (λ)
     - Clip ratio
     - Number of epochs per update
     - Batch size

2. **Training Loop**:
   - For each episode:
     a. Reset environment and get initial state
     b. Collect trajectories:
        - Sample actions using current policy
        - Execute actions and collect rewards
        - Store transitions (state, action, reward, next state)
     c. Compute advantages using GAE:
        ```
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)
        ```
     d. Update policy using clipped surrogate objective:
        ```
        L(θ) = min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)
        ```
     e. Update value function to minimize MSE
     f. Save model periodically
     g. Log performance metrics

3. **Performance Metrics**:
   - Average reward per episode
   - Policy loss
   - Value loss
   - Entropy
   - Queue lengths
   - Waiting times
   - Flow rates

## Testing and Evaluation

The `test_agent.py` script provides comprehensive testing capabilities:

1. **Experimental Design**:
   - Tests three types of traffic light control:
     1. Fixed-time (baseline)
     2. Q-Learning
     3. PPO
   - Tests each agent on multiple road networks:
     1. road1.net.xml
     2. road2.net.xml
     3. road3.net.xml
   - Runs multiple episodes (10) for statistical significance
   - Compares performance against baseline fixed-time control

2. **Performance Metrics**:
   - Average waiting time
   - Queue length
   - Throughput
   - Emergency vehicle response time
   - Statistical analysis (mean and standard deviation)
   - Comparison with fixed-time baseline

3. **Output**:
   - Visual comparison charts
   - Detailed CSV reports
   - Performance summaries
   - Statistical analysis
   - Baseline comparison results
   - Results organized by road network
