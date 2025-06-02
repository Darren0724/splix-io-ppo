import asyncio
import platform
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import collections
from tqdm import tqdm
import torch.nn as nn
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import imageio
import matplotlib.pyplot as plt # Import matplotlib

# --- Constants ---
HEAD_ID = 3
TERRITORY_ID = 1
BODY_ID = 2
EMPTY_ID = 0
WALL_ID = 4
HIDDEN_DIM = 128

NUM_GRID_CLASSES = 5  # EMPTY_ID, TERRITORY_ID, BODY_ID, HEAD_ID, WALL_ID
NUM_ACTIONS = 4       # Up, Right, Down, Left

# Shaped Reward constants
REWARD_PER_CELL_FILLED = 1.0
REWARD_PER_CELL_FILLED_SHAPED = 5.0
PENALTY_DEATH = -20.0
PENALTY_INVALID_MOVE = -1.0
PENALTY_TIME_STEP = -0.1
PENALTY_TRAIL_TOO_FAR = -0.2
REWARD_EXTEND_TRAIL_ON_EMPTY = 0.05
MAX_DISTANCE = 3
OBS_SIZE = 5
ACTUAL_GRID_SIZE = 30 # Actual grid size for the environment, larger than OBS_SIZE



FIXED_EVAL_START_POSITIONS = [
    (5,5), (5,15), (5,24), (10,8), (10,18), 
    (14,5), (14,14), (14,24), (19,8), (19,18), 
    (24,5), (24,15), (24,24), (7,10), (7,20), 
    (12,12), (12,22), (17,7), (17,17), (22,10)
] # 20 fixed positions, ensuring they are within [1, ACTUAL_GRID_SIZE-2]
FIXED_EVAL_START_DIRECTION = 0 # e.g., Up

# Custom wrapper to add channel dimension to grid observation
class GridToImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(GridToImageWrapper, self).__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("GridToImageWrapper expects the wrapped environment's observation space to be a gym.spaces.Dict.")
        if 'grid' not in env.observation_space.spaces:
            raise ValueError("The wrapped environment's observation_space Dict must contain 'grid' and 'sign_array' keys.")

        self.grid_space = env.observation_space.spaces['grid'] # Shape (H, W, C_one_hot)
        

        # max_raw_obs_val is not needed for one-hot encoded grid as values are 0 or 1
        # self.max_raw_obs_val = ... (remove or comment out this block)

        original_grid_shape = self.grid_space.shape # (H, W, C_one_hot)
        num_one_hot_channels = original_grid_shape[2]

        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0.0, high=1.0,
                shape=(num_one_hot_channels, original_grid_shape[0], original_grid_shape[1]),
                dtype=np.float32
            )
        })

        # Check H and W dimensions
        if original_grid_shape[0] != OBS_SIZE or original_grid_shape[1] != OBS_SIZE:
            raise ValueError(f"Grid shape {original_grid_shape[:2]} from wrapped env does not match expected OBS_SIZE ({OBS_SIZE},{OBS_SIZE})")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, obs_dict):
        grid_obs_one_hot = obs_dict['grid'].astype(np.float32) # Shape (H, W, C_one_hot)
        
        # Transpose from (H, W, C_one_hot) to (C_one_hot, H, W) for PyTorch CNN convention
        processed_grid = grid_obs_one_hot.transpose(2, 0, 1)
        
        return {'grid': processed_grid} # Ensure last_action is float


# SplixIOEnv (assuming this class definition is the same as your latest version with partial observation)
class SplixIOEnv(gym.Env):
    metadata = {'render_modes': ['human', 'none']} # Added 'none'

    def __init__(self, grid_size=ACTUAL_GRID_SIZE, obs_size=OBS_SIZE, render_mode='human'):
        super(SplixIOEnv, self).__init__()
        self.actual_grid_size = grid_size
        self.obs_size = obs_size
        self.render_mode = render_mode # Store render_mode

        self.action_space = spaces.Discrete(NUM_ACTIONS) # Use NUM_ACTIONS
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0, high=1, # Values will be 0 or 1 for one-hot
                shape=(self.obs_size, self.obs_size, NUM_GRID_CLASSES), # H, W, C_one_hot
                dtype=np.int32 # Or np.uint8
            )
        })

        self.grid = np.zeros((self.actual_grid_size, self.actual_grid_size), dtype=np.int32)
        self.real_grid = np.zeros((self.actual_grid_size, self.actual_grid_size), dtype=np.int32)
        
        self.player_pos = [self.actual_grid_size // 2, self.actual_grid_size // 2]
        self.player_trail = []
        self.direction = 0 # This can still store the integer direction if needed internally
        self.alive = True
        
        self.screen = None
        self.max_steps = self.actual_grid_size * self.actual_grid_size * 3 # Example max steps
        self.steps = 0
        self.clock = pygame.time.Clock()
        self.FPS = 10
        self.last_action = np.zeros((NUM_ACTIONS,), dtype=np.int32) # Initialize as one-hot (e.g., all zeros for initial state)
        # self.render_mode = render_mode # Already set

    def reset(self, seed=None, options=None, start_pos=None, start_direction=None): # Added start_pos and start_direction
        super().reset(seed=seed)
        self.grid = np.zeros((self.actual_grid_size, self.actual_grid_size), dtype=np.int32)
        self.real_grid = np.zeros((self.actual_grid_size, self.actual_grid_size), dtype=np.int32)
        self.last_action = np.zeros((NUM_ACTIONS,), dtype=np.int32) # Reset last action

        if start_pos:
            # Ensure start_pos allows 3x3 territory within bounds
            # Valid range for start_pos elements is [1, self.actual_grid_size - 2]
            px, py = start_pos
            if not (1 <= px < self.actual_grid_size -1 and 1 <= py < self.actual_grid_size -1):
                raise ValueError(f"Provided start_pos {start_pos} is out of valid range for 3x3 initial territory.")
            self.player_pos = list(start_pos) 
        else:
            # Ensure player starts on territory, at least 1 cell away from border for initial 3x3 territory
            self.player_pos = [np.random.randint(1, self.actual_grid_size - 1), 
                               np.random.randint(1, self.actual_grid_size - 1)]
        
        # Initialize 3x3 territory around player_pos
        for x_offset in range(-1, 2):
            for y_offset in range(-1, 2):
                r, c = self.player_pos[0] + x_offset, self.player_pos[1] + y_offset
                # Boundary check already implicitly handled by player_pos constraints
                self.real_grid[r, c] = TERRITORY_ID
                self.grid[r, c] = TERRITORY_ID
        
        self.grid[self.player_pos[0], self.player_pos[1]] = HEAD_ID # Head is on territory
        
        self.player_trail = [self.player_pos.copy()]

        if start_direction is not None:
            self.direction = start_direction
        else:
            self.direction = np.random.randint(0, NUM_ACTIONS) 
        
        self.alive = True
        self.steps = 0
        # if self.render_mode == 'human': # Moved render call to after state is fully reset
        #     self.render()
        return self.gen_obs(), {}

    def _calculate_distance_to_territory(self):
        player_r, player_c = self.player_pos
        territory_cells = np.argwhere(self.real_grid == TERRITORY_ID)
        if territory_cells.size == 0: return float('inf') # Should not happen if player starts on territory
        
        min_dist = float('inf')
        for ter_r, ter_c in territory_cells:
            dist = max(abs(player_r - ter_r), abs(player_c - ter_c))
            if dist < min_dist: min_dist = dist
        return min_dist

    def _fill_territory(self, trail_path_to_evaluate):
        if len(trail_path_to_evaluate) < 4: return 0 # Need at least 3 trail points + current pos to form a loop

        BOUNDARY_FILL_ID = -1 # A temporary ID for boundaries during flood fill
        temp_flood_grid = np.copy(self.real_grid) # Use real_grid as base for flood fill logic

        # Mark the trail path on the temporary grid
        for r_idx, c_idx in trail_path_to_evaluate:
            if 0 <= r_idx < self.actual_grid_size and 0 <= c_idx < self.actual_grid_size:
                temp_flood_grid[r_idx, c_idx] = BOUNDARY_FILL_ID
        
        q = collections.deque()
        visited_for_outside_fill = np.zeros_like(self.grid, dtype=bool)

        # Start flood fill from all EMPTY border cells to identify outside area
        for r_idx in range(self.actual_grid_size):
            for c_idx in [0, self.actual_grid_size - 1]: # Left and right borders
                if temp_flood_grid[r_idx, c_idx] == EMPTY_ID and not visited_for_outside_fill[r_idx,c_idx]:
                    q.append((r_idx, c_idx))
                    visited_for_outside_fill[r_idx, c_idx] = True
        for c_idx in range(1, self.actual_grid_size - 1): # Top and bottom borders (excluding corners already covered)
             for r_idx in [0, self.actual_grid_size -1]:
                if temp_flood_grid[r_idx, c_idx] == EMPTY_ID and not visited_for_outside_fill[r_idx,c_idx]:
                    q.append((r_idx, c_idx))
                    visited_for_outside_fill[r_idx, c_idx] = True
        
        while q:
            r, c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-directional
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.actual_grid_size and 0 <= nc < self.actual_grid_size and \
                   not visited_for_outside_fill[nr, nc] and temp_flood_grid[nr, nc] == EMPTY_ID:
                    visited_for_outside_fill[nr, nc] = True
                    q.append((nr, nc))
        
        filled_count = 0
        for r_fill in range(self.actual_grid_size):
            for c_fill in range(self.actual_grid_size):
                # A cell is filled if it was EMPTY on real_grid, not part of the trail boundary, and not reachable from outside
                if self.real_grid[r_fill,c_fill] == EMPTY_ID and \
                   temp_flood_grid[r_fill,c_fill] != BOUNDARY_FILL_ID and \
                   not visited_for_outside_fill[r_fill,c_fill]:
                    self.real_grid[r_fill, c_fill] = TERRITORY_ID # Update real_grid
                    self.grid[r_fill, c_fill] = TERRITORY_ID      # Update visual grid
                    filled_count += 1
        return filled_count

    def step(self, action):
        self.steps += 1
        truncated = self.steps >= self.max_steps
        reward = PENALTY_TIME_STEP 

        direction_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)} # Up, Right, Down, Left
        move = direction_map[action]
        
        # Store previous head location from player_trail for visual updates
        previous_head_visual_location = self.player_trail[-1] # This is the last confirmed position of the head

        new_pos = [self.player_pos[0] + move[0], self.player_pos[1] + move[1]]

        # Boundary collision
        if not (0 <= new_pos[0] < self.actual_grid_size and 0 <= new_pos[1] < self.actual_grid_size):
            self.alive = False
            reward += PENALTY_DEATH
            # self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = BODY_ID # Mark last trail segment
            # No need to update self.grid for new_pos as it's out of bounds
            return self.gen_obs(), reward, True, truncated, {}
        
        # Penalty for trying to reverse direction (move into the previous trail segment)
        if len(self.player_trail) >= 2 and new_pos == self.player_trail[-2]:
            reward += PENALTY_INVALID_MOVE
            # State doesn't change, return current observation
            return self.gen_obs(), reward, False, truncated, {} # Not terminated for invalid move

        # Collision with own trail (visual grid's BODY_ID)
        if self.grid[new_pos[0], new_pos[1]] == BODY_ID:
            self.alive = False
            reward += PENALTY_DEATH
            self.grid[new_pos[0], new_pos[1]] = HEAD_ID # Show head at point of collision
            # Update previous head location to body or territory based on real_grid
            if self.real_grid[previous_head_visual_location[0], previous_head_visual_location[1]] != TERRITORY_ID:
                 self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = BODY_ID
            else: # If previous was on territory, it remains territory visually
                 self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = TERRITORY_ID
            return self.gen_obs(), reward, True, truncated, {}

        # --- Successful Move ---
        self.player_pos = new_pos # Update player position
        destination_type_on_real_grid = self.real_grid[self.player_pos[0], self.player_pos[1]]

        # Logic for entering territory
        if destination_type_on_real_grid == TERRITORY_ID:
            if len(self.player_trail) > 1: # If there's a trail to close
                current_trail_for_fill = self.player_trail + [self.player_pos.copy()] # Include current pos to close loop
                filled_count = self._fill_territory(current_trail_for_fill)
                reward += filled_count * REWARD_PER_CELL_FILLED_SHAPED
                
                # Convert the entire trail that just closed into territory on both grids
                for r_trail, c_trail in current_trail_for_fill: # Use current_trail_for_fill
                    if 0 <= r_trail < self.actual_grid_size and 0 <= c_trail < self.actual_grid_size:
                        self.real_grid[r_trail, c_trail] = TERRITORY_ID
                        self.grid[r_trail, c_trail] = TERRITORY_ID
            
            self.player_trail = [self.player_pos.copy()] # Reset trail, start new one from current territory
            # Visual update for previous head location (it was on trail or territory)
            # Since we just entered territory, the previous spot becomes territory
            self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = TERRITORY_ID

        else: # Player is on EMPTY_ID or (theoretically) their own new trail segment (which is handled by BODY_ID collision)
            # Update visual grid for the previous head location
            if self.real_grid[previous_head_visual_location[0], previous_head_visual_location[1]] == TERRITORY_ID:
                self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = TERRITORY_ID
            else: # Was on trail
                self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = BODY_ID
            
            self.player_trail.append(self.player_pos.copy()) # Extend trail
            
            # Rewards/penalties for being on trail
            if destination_type_on_real_grid == EMPTY_ID: # Moving onto empty space
                distance_to_territory = self._calculate_distance_to_territory()
                if distance_to_territory > MAX_DISTANCE:
                    reward += PENALTY_TRAIL_TOO_FAR
                else:
                    reward += REWARD_EXTEND_TRAIL_ON_EMPTY
        
        # Update current player head position on the visual grid
        self.grid[self.player_pos[0], self.player_pos[1]] = HEAD_ID
        self.direction = action # Update current direction (integer)

        terminated = not self.alive or np.count_nonzero(self.real_grid == EMPTY_ID) == 0
        # if self.render_mode == 'human':
        #     self.render()
        
        # Update last_action to one-hot
        new_last_action = np.zeros((NUM_ACTIONS,), dtype=np.int32)
        new_last_action[action] = 1
        self.last_action = new_last_action
        
        obs_dict = self.gen_obs() # gen_obs will now return the one-hot encoded grid and last_action
        return obs_dict, reward, terminated, truncated, {}

    def gen_obs(self):
        obs_grid_int = np.full((self.obs_size, self.obs_size), EMPTY_ID, dtype=np.int32) # Fill with EMPTY_ID for out-of-bounds
        center_r, center_c = self.player_pos
        half_obs = self.obs_size // 2 

        # Calculate source coordinates (from the full grid)
        src_r_start = center_r - half_obs
        src_r_end = center_r + half_obs + (self.obs_size % 2) # Add 1 if obs_size is odd
        src_c_start = center_c - half_obs
        src_c_end = center_c + half_obs + (self.obs_size % 2)

        # Calculate destination coordinates (in the obs_grid)
        dst_r_start, dst_r_end = 0, self.obs_size
        dst_c_start, dst_c_end = 0, self.obs_size
        
        # for exceed bounaderies, just fill with WALL_ID
        if src_r_start < 0:
            dst_r_start = -src_r_start
            src_r_start = 0
            obs_grid_int[0:dst_r_start, :] = WALL_ID
        if src_r_end > self.actual_grid_size:
            dst_r_end = self.obs_size - (src_r_end - self.actual_grid_size)
            src_r_end = self.actual_grid_size
            obs_grid_int[dst_r_end:, :] = WALL_ID
        if src_c_start < 0:
            dst_c_start = -src_c_start
            src_c_start = 0
            obs_grid_int[:, 0:dst_c_start] = WALL_ID
        if src_c_end > self.actual_grid_size:
            dst_c_end = self.obs_size - (src_c_end - self.actual_grid_size)
            src_c_end = self.actual_grid_size
            obs_grid_int[:, dst_c_end:] = WALL_ID
        # Copy the relevant part of the grid to the observation grid
        obs_grid_int[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = self.grid[src_r_start:src_r_end, src_c_start:src_c_end]
        
        # One-hot encode the grid
        # Ensure obs_grid_int contains values from 0 to NUM_GRID_CLASSES-1
        # For example, if WALL_ID is 4, NUM_GRID_CLASSES should be 5.
        obs_grid_one_hot = (np.arange(NUM_GRID_CLASSES) == obs_grid_int[..., None]).astype(np.int32)


       

        return {"grid": obs_grid_one_hot} # Return as a dict
    def render(self):
        if self.render_mode == 'human':
            if self.screen is None:
                pygame.init()
                pygame.display.set_caption("SplixIO RL")
                # Adjust window size based on actual_grid_size and cell size (e.g., 20x20 pixels per cell)
                self.screen = pygame.display.set_mode((self.actual_grid_size * 20, self.actual_grid_size * 20))
            
            self.screen.fill((100, 100, 100)) # Background color
            
            cell_size = 20 # Or calculate based on screen size / grid size
            for r in range(self.actual_grid_size):
                for c in range(self.actual_grid_size):
                    color = (200, 200, 200) # Default for EMPTY_ID
                    if self.grid[r, c] == TERRITORY_ID: color = (0, 150, 0)    # Green for territory
                    elif self.grid[r, c] == BODY_ID:    color = (100, 100, 255) # Light blue for trail/body
                    elif self.grid[r, c] == HEAD_ID:    color = (0, 0, 200)    # Dark blue for head
                    
                    pygame.draw.rect(self.screen, color, (c * cell_size +1 , r * cell_size +1 , cell_size-2, cell_size-2)) # Small gap
            
            pygame.display.flip()
            self.clock.tick(self.FPS) # Control game speed

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

class SplixIOevalEnv(gym.Env):
    metadata = {'render_modes': ['human', 'none']} # Added 'none'

    def __init__(self, grid_size=ACTUAL_GRID_SIZE, obs_size=OBS_SIZE, render_mode='human'):
        super(SplixIOevalEnv, self).__init__()
        self.actual_grid_size = grid_size
        self.obs_size = obs_size
        self.render_mode = render_mode # Store render_mode

        self.action_space = spaces.Discrete(NUM_ACTIONS) # Use NUM_ACTIONS
        self.observation_space = spaces.Dict({
            'grid': spaces.Box(
                low=0, high=1, # Values will be 0 or 1 for one-hot
                shape=(self.obs_size, self.obs_size, NUM_GRID_CLASSES), # H, W, C_one_hot
                dtype=np.int32 # Or np.uint8
            )
        })

        self.grid = np.zeros((self.actual_grid_size, self.actual_grid_size), dtype=np.int32)
        self.real_grid = np.zeros((self.actual_grid_size, self.actual_grid_size), dtype=np.int32)
        
        self.player_pos = [self.actual_grid_size // 2, self.actual_grid_size // 2]
        self.player_trail = []
        self.direction = 0 # This can still store the integer direction if needed internally
        self.alive = True
        
        self.screen = None
        self.max_steps = self.actual_grid_size * self.actual_grid_size * 3 # Example max steps
        self.steps = 0
        self.clock = pygame.time.Clock()
        self.FPS = 10
        self.last_action = np.zeros((NUM_ACTIONS,), dtype=np.int32) # Initialize as one-hot (e.g., all zeros for initial state)
        # self.render_mode = render_mode # Already set

    def reset(self, seed=None, options=None, start_pos=None, start_direction=None): # Added start_pos and start_direction
        super().reset(seed=seed)
        self.grid = np.zeros((self.actual_grid_size, self.actual_grid_size), dtype=np.int32)
        self.real_grid = np.zeros((self.actual_grid_size, self.actual_grid_size), dtype=np.int32)
        self.last_action = np.zeros((NUM_ACTIONS,), dtype=np.int32) # Reset last action

        if start_pos:
            # Ensure start_pos allows 3x3 territory within bounds
            # Valid range for start_pos elements is [1, self.actual_grid_size - 2]
            px, py = start_pos
            if not (1 <= px < self.actual_grid_size -1 and 1 <= py < self.actual_grid_size -1):
                raise ValueError(f"Provided start_pos {start_pos} is out of valid range for 3x3 initial territory.")
            self.player_pos = list(start_pos) 
        else:
            # Ensure player starts on territory, at least 1 cell away from border for initial 3x3 territory
            self.player_pos = [np.random.randint(1, self.actual_grid_size - 1), 
                               np.random.randint(1, self.actual_grid_size - 1)]
        
        # Initialize 3x3 territory around player_pos
        for x_offset in range(-1, 2):
            for y_offset in range(-1, 2):
                r, c = self.player_pos[0] + x_offset, self.player_pos[1] + y_offset
                # Boundary check already implicitly handled by player_pos constraints
                self.real_grid[r, c] = TERRITORY_ID
                self.grid[r, c] = TERRITORY_ID
        
        self.grid[self.player_pos[0], self.player_pos[1]] = HEAD_ID # Head is on territory
        
        self.player_trail = [self.player_pos.copy()]

        if start_direction is not None:
            self.direction = start_direction
        else:
            self.direction = np.random.randint(0, NUM_ACTIONS) 
        
        self.alive = True
        self.steps = 0
        # if self.render_mode == 'human': # Moved render call to after state is fully reset
        #     self.render()
        return self.gen_obs(), {}

    def _calculate_distance_to_territory(self):
        player_r, player_c = self.player_pos
        territory_cells = np.argwhere(self.real_grid == TERRITORY_ID)
        if territory_cells.size == 0: return float('inf') # Should not happen if player starts on territory
        
        min_dist = float('inf')
        for ter_r, ter_c in territory_cells:
            dist = max(abs(player_r - ter_r),abs(player_c - ter_c))
            if dist < min_dist: min_dist = dist
        return min_dist

    def _fill_territory(self, trail_path_to_evaluate):
        if len(trail_path_to_evaluate) < 4: return 0 # Need at least 3 trail points + current pos to form a loop

        BOUNDARY_FILL_ID = -1 # A temporary ID for boundaries during flood fill
        temp_flood_grid = np.copy(self.real_grid) # Use real_grid as base for flood fill logic

        # Mark the trail path on the temporary grid
        for r_idx, c_idx in trail_path_to_evaluate:
            if 0 <= r_idx < self.actual_grid_size and 0 <= c_idx < self.actual_grid_size:
                temp_flood_grid[r_idx, c_idx] = BOUNDARY_FILL_ID
        
        q = collections.deque()
        visited_for_outside_fill = np.zeros_like(self.grid, dtype=bool)

        # Start flood fill from all EMPTY border cells to identify outside area
        for r_idx in range(self.actual_grid_size):
            for c_idx in [0, self.actual_grid_size - 1]: # Left and right borders
                if temp_flood_grid[r_idx, c_idx] == EMPTY_ID and not visited_for_outside_fill[r_idx,c_idx]:
                    q.append((r_idx, c_idx))
                    visited_for_outside_fill[r_idx, c_idx] = True
        for c_idx in range(1, self.actual_grid_size - 1): # Top and bottom borders (excluding corners already covered)
             for r_idx in [0, self.actual_grid_size -1]:
                if temp_flood_grid[r_idx, c_idx] == EMPTY_ID and not visited_for_outside_fill[r_idx,c_idx]:
                    q.append((r_idx, c_idx))
                    visited_for_outside_fill[r_idx, c_idx] = True
        
        while q:
            r, c = q.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 4-directional
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.actual_grid_size and 0 <= nc < self.actual_grid_size and \
                   not visited_for_outside_fill[nr, nc] and temp_flood_grid[nr, nc] == EMPTY_ID:
                    visited_for_outside_fill[nr, nc] = True
                    q.append((nr, nc))
        
        filled_count = 0
        for r_fill in range(self.actual_grid_size):
            for c_fill in range(self.actual_grid_size):
                # A cell is filled if it was EMPTY on real_grid, not part of the trail boundary, and not reachable from outside
                if self.real_grid[r_fill,c_fill] == EMPTY_ID and \
                   temp_flood_grid[r_fill,c_fill] != BOUNDARY_FILL_ID and \
                   not visited_for_outside_fill[r_fill,c_fill]:
                    self.real_grid[r_fill, c_fill] = TERRITORY_ID # Update real_grid
                    self.grid[r_fill, c_fill] = TERRITORY_ID      # Update visual grid
                    filled_count += 1
        return filled_count

    def step(self, action):
        self.steps += 1
        truncated = self.steps >= self.max_steps
        reward = PENALTY_TIME_STEP 

        direction_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)} # Up, Right, Down, Left
        move = direction_map[action]
        
        # Store previous head location from player_trail for visual updates
        previous_head_visual_location = self.player_trail[-1] # This is the last confirmed position of the head

        new_pos = [self.player_pos[0] + move[0], self.player_pos[1] + move[1]]

        # Boundary collision
        if not (0 <= new_pos[0] < self.actual_grid_size and 0 <= new_pos[1] < self.actual_grid_size):
            self.alive = False
            # No need to update self.grid for new_pos as it's out of bounds
            return self.gen_obs(), reward, True, truncated, {}
        
        # Penalty for trying to reverse direction (move into the previous trail segment)
        if len(self.player_trail) >= 2 and new_pos == self.player_trail[-2]:
            reward += PENALTY_INVALID_MOVE
            # State doesn't change, return current observation
            return self.gen_obs(), reward, False, truncated, {} # Not terminated for invalid move

        # Collision with own trail (visual grid's BODY_ID)
        if self.grid[new_pos[0], new_pos[1]] == BODY_ID:
            self.alive = False
            self.grid[new_pos[0], new_pos[1]] = HEAD_ID # Show head at point of collision
            # Update previous head location to body or territory based on real_grid
            if self.real_grid[previous_head_visual_location[0], previous_head_visual_location[1]] != TERRITORY_ID:
                 self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = BODY_ID
            else: # If previous was on territory, it remains territory visually
                 self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = TERRITORY_ID
            return self.gen_obs(), reward, True, truncated, {}

        # --- Successful Move ---
        self.player_pos = new_pos # Update player position
        destination_type_on_real_grid = self.real_grid[self.player_pos[0], self.player_pos[1]]

        # Logic for entering territory
        if destination_type_on_real_grid == TERRITORY_ID:
            if len(self.player_trail) > 1: # If there's a trail to close
                current_trail_for_fill = self.player_trail + [self.player_pos.copy()] # Include current pos to close loop
                filled_count = self._fill_territory(current_trail_for_fill)
                reward += filled_count * REWARD_PER_CELL_FILLED
                
                # Convert the entire trail that just closed into territory on both grids
                for r_trail, c_trail in current_trail_for_fill: # Use current_trail_for_fill
                    if 0 <= r_trail < self.actual_grid_size and 0 <= c_trail < self.actual_grid_size:
                        self.real_grid[r_trail, c_trail] = TERRITORY_ID
                        self.grid[r_trail, c_trail] = TERRITORY_ID
            
            self.player_trail = [self.player_pos.copy()] # Reset trail, start new one from current territory
            # Visual update for previous head location (it was on trail or territory)
            # Since we just entered territory, the previous spot becomes territory
            self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = TERRITORY_ID

        else: # Player is on EMPTY_ID or (theoretically) their own new trail segment (which is handled by BODY_ID collision)
            # Update visual grid for the previous head location
            if self.real_grid[previous_head_visual_location[0], previous_head_visual_location[1]] == TERRITORY_ID:
                self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = TERRITORY_ID
            else: # Was on trail
                self.grid[previous_head_visual_location[0], previous_head_visual_location[1]] = BODY_ID
            
            self.player_trail.append(self.player_pos.copy()) # Extend trail
        
        # Update current player head position on the visual grid
        self.grid[self.player_pos[0], self.player_pos[1]] = HEAD_ID
        self.direction = action # Update current direction (integer)

        terminated = not self.alive or np.count_nonzero(self.real_grid == EMPTY_ID) == 0
        # if self.render_mode == 'human':
        #     self.render()
        
        # Update last_action to one-hot
        new_last_action = np.zeros((NUM_ACTIONS,), dtype=np.int32)
        new_last_action[action] = 1
        self.last_action = new_last_action
        
        obs_dict = self.gen_obs() # gen_obs will now return the one-hot encoded grid and last_action
        return obs_dict, reward, terminated, truncated, {}

    def gen_obs(self):
        obs_grid_int = np.full((self.obs_size, self.obs_size), EMPTY_ID, dtype=np.int32) # Fill with EMPTY_ID for out-of-bounds
        center_r, center_c = self.player_pos
        half_obs = self.obs_size // 2 

        # Calculate source coordinates (from the full grid)
        src_r_start = center_r - half_obs
        src_r_end = center_r + half_obs + (self.obs_size % 2) # Add 1 if obs_size is odd
        src_c_start = center_c - half_obs
        src_c_end = center_c + half_obs + (self.obs_size % 2)

        # Calculate destination coordinates (in the obs_grid)
        dst_r_start, dst_r_end = 0, self.obs_size
        dst_c_start, dst_c_end = 0, self.obs_size
        
        # for exceed bounaderies, just fill with WALL_ID
        if src_r_start < 0:
            dst_r_start = -src_r_start
            src_r_start = 0
            obs_grid_int[0:dst_r_start, :] = WALL_ID
        if src_r_end > self.actual_grid_size:
            dst_r_end = self.obs_size - (src_r_end - self.actual_grid_size)
            src_r_end = self.actual_grid_size
            obs_grid_int[dst_r_end:, :] = WALL_ID
        if src_c_start < 0:
            dst_c_start = -src_c_start
            src_c_start = 0
            obs_grid_int[:, 0:dst_c_start] = WALL_ID
        if src_c_end > self.actual_grid_size:
            dst_c_end = self.obs_size - (src_c_end - self.actual_grid_size)
            src_c_end = self.actual_grid_size
            obs_grid_int[:, dst_c_end:] = WALL_ID
        # Copy the relevant part of the grid to the observation grid
        obs_grid_int[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = self.grid[src_r_start:src_r_end, src_c_start:src_c_end]
        
        # One-hot encode the grid
        # Ensure obs_grid_int contains values from 0 to NUM_GRID_CLASSES-1
        # For example, if WALL_ID is 4, NUM_GRID_CLASSES should be 5.
        obs_grid_one_hot = (np.arange(NUM_GRID_CLASSES) == obs_grid_int[..., None]).astype(np.int32)


        

        return {"grid": obs_grid_one_hot} # Return as a dict
    def render(self):
        if self.render_mode == 'human':
            if self.screen is None:
                pygame.init()
                pygame.display.set_caption("SplixIO RL")
                # Adjust window size based on actual_grid_size and cell size (e.g., 20x20 pixels per cell)
                self.screen = pygame.display.set_mode((self.actual_grid_size * 20, self.actual_grid_size * 20))
            
            self.screen.fill((100, 100, 100)) # Background color
            
            cell_size = 20 # Or calculate based on screen size / grid size
            for r in range(self.actual_grid_size):
                for c in range(self.actual_grid_size):
                    color = (200, 200, 200) # Default for EMPTY_ID
                    if self.grid[r, c] == TERRITORY_ID: color = (0, 150, 0)    # Green for territory
                    elif self.grid[r, c] == BODY_ID:    color = (100, 100, 255) # Light blue for trail/body
                    elif self.grid[r, c] == HEAD_ID:    color = (0, 0, 200)    # Dark blue for head
                    
                    pygame.draw.rect(self.screen, color, (c * cell_size +1 , r * cell_size +1 , cell_size-2, cell_size-2)) # Small gap
            
            pygame.display.flip()
            self.clock.tick(self.FPS) # Control game speed

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None