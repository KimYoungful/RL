"""评估脚本：加载模型并评估若干回合。"""

import argparse
from pathlib import Path
import sys
from stable_baselines3 import SAC, PPO

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.env import EnvConfig, ArmConstraintEnv, BaseRobotEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--env-type", default="arm", choices=["arm", "base"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--algo", type=str, default="SAC", choices=["SAC", "PPO"])
    args = parser.parse_args()

    env_cls = ArmConstraintEnv if args.env_type == "arm" else BaseRobotEnv
    env_cfg = EnvConfig.from_yaml(args.env_config)
    env = env_cls(config=env_cfg)

    algo = args.algo.upper()
    if algo == "SAC":
        model = SAC.load(args.model_path, env=env)
    else:
        model = PPO.load(args.model_path, env=env)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, truncated, info = env.step(action)
            ep_rew += rew
        print(f"Episode {ep+1}: reward={ep_rew:.2f}")
    env.close()


if __name__ == "__main__":
    main()


