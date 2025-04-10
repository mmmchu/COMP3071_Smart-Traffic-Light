import torch
import torch.nn as nn


class QlAgent:
    def __init__(self, input_shape, output_shape=8, learning_rate=1e-3, epsilon=1):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.input_shape = input_shape  # Dynamically set input shape
        self.output_shape = output_shape

        # Define neural network
        self.model = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_shape),
        )

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def predict_rewards(self, observations):
        """Predict rewards while dynamically adjusting input size."""
        obs_size = observations.shape[0]  # Get actual input size

        if obs_size != self.input_shape:
            if obs_size < self.input_shape:
                # Pad with -1 (or 0) if the input is smaller
                observations = torch.cat([observations, torch.full((self.input_shape - obs_size,), -1.0)])
            else:
                # Trim if the input is larger
                observations = observations[:self.input_shape]

        return self.model(observations)

    def learn(self, pred_reward, actual_reward, emergency_present):
        """Perform backpropagation and update the model with emergency handling."""
        self.optimizer.zero_grad()
        loss = self.loss_fn(pred_reward, actual_reward)

        # **Increase penalty if emergency vehicle is waiting**
        if emergency_present:
            loss *= 1.5  # Increase loss impact to prioritize emergency clearance

        loss.backward()
        self.optimizer.step()

    def save_model(self, filename="trained_model.pth"):
        """Save the trained model."""
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename="trained_model.pth"):
        """Load a pre-trained model."""
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Set to evaluation mode
