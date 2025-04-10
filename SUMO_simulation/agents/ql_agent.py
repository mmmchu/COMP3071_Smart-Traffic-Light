import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, Any

class QlAgent:
    def __init__(self, input_shape: int, output_shape: int, learning_rate: float = 1e-3, epsilon: float = 1.0, batch_size: int = 32, memory_size: int = 10000):
        """
        Initialize Q-learning agent.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.memory_size = memory_size
        
        # Initialize neural network
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_shape)
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
        
        # Initialize reward normalization buffer
        self.reward_buffer = deque(maxlen=1000)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0
    
    def predict_rewards(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict Q-values for a given state.
        """
        with torch.no_grad():
            return self.model(state)
    
    def store_experience(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool, emergency_present: bool = False):
        """
        Store experience in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done, emergency_present))
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay memory (legacy method).
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(next_state, np.ndarray):
            next_state = torch.from_numpy(next_state).float()
            
        # Add emergency_present as False for backward compatibility
        self.store_experience(state, action, reward, next_state, done, False)
    
    def learn(self, pred_q_values: torch.Tensor, reward: float, emergency_present: bool = False):
        """
        Learn from experience using experience replay.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, emergencies = zip(*batch)
        
        # Convert to tensors
        states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in states])
        next_states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in next_states])
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        emergencies = torch.tensor(emergencies, dtype=torch.float32)
        
        # Get current Q values
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.model(next_states)
            next_q_values = next_q_values.max(1)[0]
            next_q_values[dones] = 0.0  # Set to 0 for terminal states
        
        # Compute target Q values
        target_q_values = rewards + 0.99 * next_q_values
        
        # Adjust target Q values for emergency vehicles
        target_q_values = torch.where(emergencies == 1, target_q_values * 1.5, target_q_values)
        
        # Compute loss and update model
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, path: str):
        """
        Save model to disk.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_count': self.reward_count
        }, path)
    
    def load_model(self, path: str):
        """
        Load model from disk.
        """
        state = torch.load(path)
        self.model.load_state_dict(state['model_state_dict'])
        if 'optimizer_state_dict' in state:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        if 'reward_mean' in state:
            self.reward_mean = state['reward_mean']
        if 'reward_std' in state:
            self.reward_std = state['reward_std']
        if 'reward_count' in state:
            self.reward_count = state['reward_count']

    def adjust_model(self, input_shape: int, output_shape: int) -> None:
        """
        Adjust the model's input and output shapes dynamically.
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_shape)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] or None:
        """
        Sample a batch of experiences from memory.
        """
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = [exp[0] for exp in batch]
        actions = [exp[1] for exp in batch]
        rewards = [exp[2] for exp in batch]
        next_states = [exp[3] for exp in batch]
        dones = [exp[4] for exp in batch]
        emergencies = [exp[5] for exp in batch]
        
        # Convert to tensors
        states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in states])
        next_states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in next_states])
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        emergencies = torch.tensor(emergencies, dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones, emergencies
    
    def normalize_reward(self, reward: float) -> float:
        """
        Normalize reward using a fixed-size buffer.
        """
        self.reward_buffer.append(reward)
        self.reward_mean = np.mean(self.reward_buffer)
        self.reward_std = np.std(self.reward_buffer) + 1e-8  # Avoid division by zero
        self.reward_count = len(self.reward_buffer)
        return (reward - self.reward_mean) / self.reward_std
    
    def process_state(self, state):
        """Process state into the correct format for the neural network."""
        if isinstance(state, dict):
            # For multi-agent case, concatenate all observations
            state = np.concatenate([obs for obs in state.values()])
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif isinstance(state, list):
            state = torch.tensor(state, dtype=torch.float32)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        return state