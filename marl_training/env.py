import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
import sys
import os

# Import the shared contracts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.contracts import (
    OBS_DIM, ACTION_DIM, MAX_XY_VELOCITY, MAX_Z_VELOCITY, 
    MAX_YAW_RATE, COMMS_RADIUS, MAX_ALTITUDE, MAP_EXTENT, 
    RawDroneState, build_observation_vector
)

class DroneSwarmEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "drone_swarm_v1"}

    def __init__(self, num_drones=3, max_steps=100):
        self.possible_agents = [f"drone_{i}" for i in range(1, num_drones + 1)]
        self.agents = self.possible_agents.copy()
        self.max_steps = max_steps
        self.dt = 0.1 
        self.target_position = np.array([40.0, 40.0, 5.0])
        
        self.observation_spaces = {agent: Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32) for agent in self.possible_agents}
        self.action_spaces = {agent: Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32) for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.states = {}
        for i, agent in enumerate(self.agents):
            self.states[agent] = RawDroneState(
                drone_id=agent,
                position=np.array([float(i * 2), 0.0, 2.0]), 
                velocity=np.zeros(3),
                yaw=0.0
            )
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self.current_step += 1
        
        for agent, action in actions.items():
            state = self.states[agent]
            cmd_vx = np.clip(action[0], -1.0, 1.0) * MAX_XY_VELOCITY
            cmd_vy = np.clip(action[1], -1.0, 1.0) * MAX_XY_VELOCITY
            cmd_vz = np.clip(action[2], -1.0, 1.0) * MAX_Z_VELOCITY
            cmd_yaw_rate = np.clip(action[3], -1.0, 1.0) * MAX_YAW_RATE
            
            state.velocity = np.array([cmd_vx, cmd_vy, cmd_vz])
            state.position += state.velocity * self.dt
            state.position = np.clip(state.position, 0.0, MAP_EXTENT) 
            state.yaw += cmd_yaw_rate * self.dt

        rewards, terminations, truncations, infos = {}, {}, {}, {}
        all_drones = list(self.states.values())
        
        for agent in self.agents:
            state = self.states[agent]
            dist_to_target = np.linalg.norm(self.target_position - state.position)
            
            # Base penalty for taking time (encourages speed)
            reward = -0.1 - (dist_to_target * 0.01) 
            
            terminated = False
            for other_state in all_drones:
                if state.drone_id != other_state.drone_id:
                    if np.linalg.norm(state.position - other_state.position) < 0.5: 
                        reward -= 100.0 # Crash penalty
                        terminated = True
            
            if dist_to_target < 1.0:
                reward += 100.0 # Success reward
                terminated = True

            rewards[agent] = reward
            terminations[agent] = terminated
            truncations[agent] = self.current_step >= self.max_steps
            infos[agent] = {}

        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]
        observations = {agent: self._get_obs(agent) if agent in self.agents else np.zeros(OBS_DIM, dtype=np.float32) for agent in actions.keys()}
        
        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent_id):
        self_state = self.states[agent_id]
        neighbors = [s for k, s in self.states.items() if k != agent_id]
        return build_observation_vector(self_state, self.target_position, neighbors)

    def render(self):
        pass
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    def action_space(self, agent):
        return self.action_spaces[agent]
