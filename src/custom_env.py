import gymnasium as gym
from gymnasium.spaces import Box, Discrete,Tuple
import numpy as np
import pygame
import time

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)  # Color for the trajectory

class CustomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,render_mode=None):
        super().__init__()
        self.grid_size = 10
        self.pause = False
        self.domain_randomization = False
        self.render_mode = render_mode

        # Define two separate thresholds for obstacle handling
        self.distance_threshold_penalty = 3  # Penalty zone threshold (larger value)
        self.distance_threshold_collision = 1.5  # Collision threshold (smaller value)
        self.penalty_factor = 5  # Penalty scaling factor
        self.distance_reward_factor = 10
        self.smooth_action_penalty = 1
        self.steps = 0
        self.margin = 0.1


        self.current_distance = 0  # Current distance to goal, used for reward shaping
        self.max_steps = 30  # Set a maximum number of steps to prevent infinite loops
        # Action space (dx, dy)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # Observation space (robot_x, robot_y, goal_x, goal_y)
        self.observation_shape = 2+2+2+1+1  # Robot position, hand position, velocity_hand,radius_hand, and distance to hand
        self.observation_space = Box(low=0, high=np.array([self.grid_size,self.grid_size, self.grid_size, self.grid_size,1,1 , (2**0.5)*self.grid_size,0.5*self.grid_size]), 
                                    shape=(self.observation_shape,), dtype=np.float32)

        self.random = True
        # For rendering
        self.window = None
        self.clock = None
        self.cell_size = 50 # Pixels per grid unit
        self.trajectory_points = [] # New: List to store past robot positions

    def _get_obs(self):
        
        return np.concatenate(([self.robot_position]+ [self.hand_position]+[self.last_action]+
                               [np.array([self.current_distance])]+
                               [np.array([min(self.robot_position[0],
                                              self.robot_position[1],
                                              self.grid_size-self.robot_position[0],
                                              self.grid_size-self.robot_position[1]) ])]))

    def _get_info(self):
        return {
            "distance_to_hand": self.current_distance,
            "robot_position": self.robot_position,
            "hand_position": self.hand_position,
        }

    def reset(self, seed=None, options=None):

        super().reset()
        self.stride_hand = np.random.uniform(0.6, 1)  # Randomize stride length
        self.distance_threshold_collision = np.random.uniform(1, 2)  # Randomize collision threshold
        self.distance_threshold_penalty = np.random.uniform(2, 4)  # Randomize penalty threshold
        
        
        self.noise_obs_sigma = np.random.uniform(0, 0.1)  # Add some noise to observation to make it more realistic
        self.noise_action_sigma = np.random.uniform(0,0.1)  # Add some noise to action to make it more realistic
        
        
        
        self.robot_position = np.random.uniform(self.margin, self.grid_size-self.margin, size=2)
        self.hand_position = np.random.uniform(self.margin, self.grid_size-self.margin, size=2)
        # self.hand_position = np.clip(self.hand_position, self.margin, self.grid_size-self.margin)  # Ensure hand stays within grid bounds
        
        self.hand_move_mode = 'random' if np.random.rand() < 0.1 else 'towards_robot'  # Randomize hand movement mode
        
        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)
        self.pre_distance = self.current_distance
        self.last_action = np.zeros(2)
        self.steps = 0
        self.trajectory_points = [self.robot_position.copy()] # New: Reset trajectory and add initial position
        
        return self._get_obs(), self._get_info()

    def _reward(self,action):
        terminated = False
        truncated = False
        reward = 0  # Initialize reward
        done_reason = None  # Initialize done reason

        # action regulation penalty

        reward -= 0.5 * np.sum(np.square(action))  # Penalty for large actions

        if np.any(self.robot_position <= self.margin) or np.any(self.grid_size-self.robot_position <=self.margin):
            reward -= 400
            terminated = True  # Truncate if robot goes out of bounds
            done_reason = "out of bounds"

        # self.robot_position = np.clip(self.robot_position, self.margin, self.grid_size-self.margin)  # Ensure robot stays within grid bounds
        # boundary penalty

        reward -= self.smooth_action_penalty*np.linalg.norm(action-self.last_action)  # Penalty for smooth action
        self.last_action = action
        # Auxiliary Rewards -  distance to hand
        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)
        reward += (self.current_distance-self.pre_distance)*self.distance_reward_factor  # Reward shaping based on distance change
        self.pre_distance = self.current_distance

        # Obstacle handling with two thresholds
        if self.current_distance < self.distance_threshold_collision:
            reward -= 100
            terminated = True  # Terminate if too close to obstacles
            done_reason = "collision with obstacle"
        elif self.current_distance < self.distance_threshold_penalty:
            reward -= self.penalty_factor * (self.distance_threshold_penalty - self.current_distance)  # Penalty for being too close to obstacles

        
        
        reward+=  4  # Small reward for each step taken to encourage exploration
        if self.steps >= self.max_steps:
            reward += 50
            truncated = True  # Truncate if max steps reached
        return reward,terminated,truncated,done_reason

    def _get_hand_movement(self):

        if self.hand_move_mode == 'random':
            move_hand = np.random.uniform(-1, 1, size=2)  # Randomly move the hand position slightly
        elif self.hand_move_mode == 'towards_robot':
            dir_vector = self.robot_position - self.hand_position
            if np.linalg.norm(dir_vector) > 0:
                dir_vector /= np.linalg.norm(dir_vector)
            move_hand = dir_vector * self.stride_hand  # Move hand towards robot position
        
        return move_hand






    def step(self, action):
        if self.random:
            action+=np.random.normal(0,self.noise_action_sigma,size=self.action_space.shape)  # Add some noise to action to make it more realistic

        move_hand = self._get_hand_movement()
        self.hand_position += move_hand  # Update hand position
        self.hand_position = np.clip(self.hand_position, self.margin, self.grid_size-self.margin)  # Ensure hand stays within grid bounds
        stride = 1  # Base stride length
        self.robot_position += action * stride  # Scale the action to control speed
        self.trajectory_points.append(self.robot_position.copy()) # New: Add current position to trajectory
        self.steps += 1

        reward,terminated,truncated,done_reason = self._reward(action)
        info = self._get_info()
        info['done_reason'] = done_reason
        observation = self._get_obs()
        if self.random:
            observation += np.random.normal(0, self.noise_obs_sigma, size=self.observation_shape)  # Add some noise to observation to make it more realistic

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
 
        pygame.display.init()
        self.window = pygame.display.set_mode(
                (int(self.grid_size * self.cell_size), int(self.grid_size * self.cell_size))
            )
        pygame.display.set_caption("CustomEnv")
        if self.clock is None:
            self.clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit() # Exit the program

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                self.hand_position = np.array([mouse_x/self.cell_size, mouse_y/self.cell_size])

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:  # 空格键切换暂停
                    self.pause = not self.pause


        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        canvas.fill(WHITE)
        virus_image = pygame.image.load("hand.png").convert_alpha()  # Load an image if needed, but not used here
        robot_image = pygame.transform.scale(virus_image, (int(self.cell_size * 2), int(self.cell_size * 2)))  # Scale the image
        # New: Draw the trajectory
        if len(self.trajectory_points) > 1:
            scaled_points = []
            for point in self.trajectory_points:
                scaled_points.append((int(point[0] * self.cell_size), int(point[1] * self.cell_size)))
            
            # Draw lines between consecutive points
            pygame.draw.lines(canvas, BLUE, False, scaled_points, 2) # Blue line, not closed, 2 pixels wide
            
            # Optionally, draw small circles at each point to emphasize
            for point_coord in scaled_points:
                pygame.draw.circle(canvas, BLUE, point_coord, 3) # Small blue circles

        # Draw robot
        pygame.draw.circle(
            canvas,
            RED,
            (int(self.robot_position[0] * self.cell_size), int(self.robot_position[1] * self.cell_size)),
            int(self.cell_size * 0.2)
        )
        # Draw obstacles

        canvas.blit(robot_image, (int((self.hand_position[0]-1) * self.cell_size), int((self.hand_position[1]-1) * self.cell_size+1)))
        pygame.draw.circle(canvas,
                            GREEN, 
                            (int((self.hand_position[0]) * self.cell_size), 
                            int((self.hand_position[1]) * self.cell_size+1)), 
        int(self.cell_size * 0.2)
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        time.sleep(0.5)

    def close(self):
        pygame.display.quit()
        pygame.quit()



