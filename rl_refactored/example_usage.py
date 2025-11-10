"""使用示例"""

from src.env import ArmConstraintEnv, EnvConfig
from src.training import TrainingConfig, Trainer
from pathlib import Path

# 示例1: 创建环境并测试
def example_create_env():
    """创建环境示例"""
    # 加载配置
    config = EnvConfig.from_yaml("config/env/arm_constraint.yaml")
    
    # 创建环境
    env = ArmConstraintEnv(config=config)
    
    # 重置环境
    obs, info = env.reset()
    print(f"观测维度: {obs.shape}")
    print(f"信息: {info}")
    
    # 执行几步
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"步骤 {i+1}: 奖励={reward:.2f}, 完成={done}, 截断={truncated}")
        if done or truncated:
            obs, info = env.reset()
    
    env.close()

# 示例2: 训练模型
def example_train():
    """训练模型示例"""
    # 加载配置
    env_config = EnvConfig.from_yaml("config/env/arm_constraint.yaml")
    training_config = TrainingConfig.from_yaml("config/training/sac.yaml")
    
    # 创建训练器
    log_dir = Path("logs/example_training")
    trainer = Trainer(
        env_config=env_config,
        training_config=training_config,
        log_dir=log_dir,
        env_class=ArmConstraintEnv
    )
    
    # 开始训练（这里只训练少量步数作为示例）
    training_config.total_timesteps = 10000  # 示例：只训练10000步
    trainer.train()

if __name__ == "__main__":
    print("示例1: 创建环境")
    example_create_env()
    
    print("\n示例2: 训练模型（取消注释以运行）")
    # example_train()  # 取消注释以运行训练示例

