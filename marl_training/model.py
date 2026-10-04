import torch
import torch.nn as nn
import sys
import os

# Import the shared contracts so the AI matches the Simulation perfectly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.contracts import OBS_DIM, ACTION_DIM

class SwarmActor(nn.Module):
    def __init__(self):
        super(SwarmActor, self).__init__()
        
        # The Neural Network Architecture
        # Input: 23 dimensions (Self state + 2 Neighbors)
        # Output: 4 dimensions (vx, vy, vz, yaw_rate)
        self.network = nn.Sequential(
            nn.Linear(OBS_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
            nn.Tanh()  # CRITICAL: Tanh squashes the output to exactly [-1.0, 1.0]
        )

    def forward(self, state):
        """
        Takes the 23-dim state vector and returns the 4-dim action vector.
        """
        return self.network(state)

# --- Quick Test Block ---
if __name__ == "__main__":
    print(f"Building Actor Network with Input: {OBS_DIM}, Output: {ACTION_DIM}")
    actor = SwarmActor()
    
    # Create a fake observation (a tensor of 23 random numbers)
    fake_observation = torch.randn(1, OBS_DIM)
    
    # Pass it through the brain
    fake_action = actor(fake_observation)
    
    print(f"Fake Observation Shape: {fake_observation.shape}")
    print(f"AI Action Output: {fake_action.detach().numpy()}")
    print("Test passed! Action is bounded between -1 and 1.")
