"""训练脚本"""

import argparse
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.env import EnvConfig, ArmConstraintEnv, BaseRobotEnv
from src.training import TrainingConfig, Trainer


def main():
    parser = argparse.ArgumentParser(description="训练强化学习模型")
    parser.add_argument(
        "--env-config",
        type=str,
        required=True,
        help="环境配置文件路径"
    )
    parser.add_argument(
        "--training-config",
        type=str,
        required=True,
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/training",
        help="日志目录"
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="arm",
        choices=["arm", "base"],
        help="环境类型: arm (带手臂约束) 或 base (基础环境)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    env_config = EnvConfig.from_yaml(args.env_config)
    training_config = TrainingConfig.from_yaml(args.training_config)
    
    # 选择环境类
    env_class = ArmConstraintEnv if args.env_type == "arm" else BaseRobotEnv
    
    # 创建训练器
    log_dir = Path(args.log_dir)
    trainer = Trainer(
        env_config=env_config,
        training_config=training_config,
        log_dir=log_dir,
        env_class=env_class
    )
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()

