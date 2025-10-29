import numpy as np
import time
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from src.custom_env import CustomEnv
from renderer import render_environment

def main():
    env = DummyVecEnv([lambda: VecMonitor(CustomEnv())])  # Wrap the environment with Monitor for logging
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.0008, ent_coef=0.02, seed=42)
    
    # Load the best model if available
    model = SAC.load("logs/best_model_sac1/best_model.zip", env=env)
    
    obs, _ = env.reset()
    print(obs)
    
    for i in range(1000000):
        obs = env._get_obs()
        action, _states = model.predict(obs, deterministic=True)

        obs, reward, terminated, _, info = env.step(action)
        print(f"obs: {obs}")
        print(action, info["distance_to_hand"])
        print(info["robot_position"], info["hand_position"])
        print("Reward:", reward)

        # Render the environment using the new rendering module
        render_environment(info["robot_position"], info["hand_position"], env.trajectory_points)

        time.sleep(0.6)  # Control the frame rate
        if terminated:
            env.render()
            time.sleep(0.6)
            env.close()
            break

if __name__ == "__main__":
    main()