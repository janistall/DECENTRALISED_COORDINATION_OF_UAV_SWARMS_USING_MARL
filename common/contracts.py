import numpy as np
from dataclasses import dataclass

MAX_XY_VELOCITY = 3.0       # m/s
MAX_Z_VELOCITY = 1.5        # m/s
MAX_YAW_RATE = np.pi / 2    # rad/s
COMMS_RADIUS = 15.0         # meters
MAX_ALTITUDE = 10.0         # meters
MAP_EXTENT = 50.0           # meters
OBS_DIM = 23
ACTION_DIM = 4

@dataclass
class RawDroneState:
    drone_id: str
    position: np.ndarray    # [x, y, z]
    velocity: np.ndarray    # [vx, vy, vz]
    yaw: float
    is_active: bool = True

def scale_action(norm_action: np.ndarray) -> np.ndarray:
    """Scales [-1, 1] AI output to physical limits."""
    c = np.clip(norm_action, -1.0, 1.0)
    return np.array([
        c[0] * MAX_XY_VELOCITY, c[1] * MAX_XY_VELOCITY, 
        c[2] * MAX_Z_VELOCITY, c[3] * MAX_YAW_RATE
    ], dtype=np.float32)

# (Observation builder logic follows rules defined in 3.3)
