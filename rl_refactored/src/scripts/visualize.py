"""可视化脚本（占位）：渲染环境运行。"""

import argparse
from pathlib import Path
import sys
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.env import EnvConfig, ArmConstraintEnv, BaseRobotEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--env-type", default="arm", choices=["arm", "base"])
    args = parser.parse_args()

    env_cls = ArmConstraintEnv if args.env_type == "arm" else BaseRobotEnv
    env_cfg = EnvConfig.from_yaml(args.env_config)
    env = env_cls(config=env_cfg)

    obs, info = env.reset()
    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)
        if done or truncated:
            obs, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()


