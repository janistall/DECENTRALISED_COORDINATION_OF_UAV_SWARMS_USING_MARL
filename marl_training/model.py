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

class SwarmCritic(nn.Module):
    def __init__(self, num_drones=3):
        super(SwarmCritic, self).__init__()
        
        # Centralized Training: The Critic sees EVERYTHING (all states + all actions)
        total_obs = OBS_DIM * num_drones
        total_actions = ACTION_DIM * num_drones
        
        self.network = nn.Sequential(
            nn.Linear(total_obs + total_actions, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Outputs a single Q-value (the "score")
        )

    def forward(self, state_all, action_all):
        """
        state_all: Concatenated 23-dim states of all 3 drones (69 dims)
        action_all: Concatenated 4-dim actions of all 3 drones (12 dims)
        """
        # Glue the states and actions together into one giant list
        x = torch.cat([state_all, action_all], dim=1)
        return self.network(x)
# --- Quick Test Block ---
if __name__ == "__main__":
    print("--- Testing Actor ---")
    actor = SwarmActor()
    fake_obs = torch.randn(1, OBS_DIM)
    fake_action = actor(fake_obs)
    print(f"Actor Output Shape: {fake_action.shape} (Should be [1, {ACTION_DIM}])")
    
    print("\n--- Testing Centralized Critic ---")
    critic = SwarmCritic(num_drones=3)
    
    # Fake data for 3 drones
    fake_all_states = torch.randn(1, OBS_DIM * 3)
    fake_all_actions = torch.randn(1, ACTION_DIM * 3)
    
    score = critic(fake_all_states, fake_all_actions)
    print(f"Critic Score Output: {score.detach().numpy()}")
    print("Test passed! Both networks are ready for MADDPG.")
