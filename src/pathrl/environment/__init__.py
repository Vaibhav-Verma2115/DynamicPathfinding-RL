import gymnasium as gym

from pathrl.environment.config import EnvConfig, RewardConfig
from pathrl.environment.grid_world import GridWorldEnv
from pathrl.environment.map_generator import GridMap, MapGenerationError, generate_map

# Allows gym.make("pathrl/GridWorld-v0", config=EnvConfig(...))
if "pathrl/GridWorld-v0" not in gym.registry:
    gym.register(id="pathrl/GridWorld-v0", entry_point="pathrl.environment.grid_world:GridWorldEnv")

__all__ = ["EnvConfig", "RewardConfig", "GridWorldEnv", "GridMap", "MapGenerationError", "generate_map"]
