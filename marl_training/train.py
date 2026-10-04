import torch
import numpy as np
from env import DroneSwarmEnv
from model import SwarmActor, SwarmCritic

def main():
    print("Initializing Simulation Environment...")
    env = DroneSwarmEnv(num_drones=3, max_steps=100)
    
    print("Initializing AI Agents...")
    # Each drone gets its own decentralized Actor brain
    actors = {agent: SwarmActor() for agent in env.possible_agents}
    
    # We only need ONE Centralized Critic to grade them all during training
    critic = SwarmCritic(num_drones=3)
    
    # Reset the environment to start the episode
    observations, infos = env.reset()
    
    print("\n--- Starting Test Flight Episode ---")
    
    episode_reward = 0
    step = 0
    
    # Run until the environment says we are done
    while env.agents:
        step += 1
        actions = {}
        
        # 1. Decentralized Execution: Each drone looks at its own state and picks an action
        for agent in env.agents:
            obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0)
            
            # Disable gradient calculation for this test run (faster)
            with torch.no_grad():
                action_tensor = actors[agent](obs_tensor)
            
            # Convert PyTorch tensor back to a standard numpy array for the environment
            actions[agent] = action_tensor.squeeze(0).numpy()
            
        # 2. Step the Environment: Apply the actions and see what happens
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Calculate total swarm reward for this step
        step_reward = sum(rewards.values())
        episode_reward += step_reward
        
        # Update observations for the next loop
        observations = next_observations
        
        if step % 20 == 0:
            print(f"Step {step}: Total Swarm Reward so far = {episode_reward:.2f}")

    print(f"\nEpisode Finished! Total Steps: {step}")
    print(f"Final Swarm Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    main()
