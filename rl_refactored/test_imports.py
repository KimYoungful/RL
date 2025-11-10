"""测试导入"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.env import EnvConfig, BaseRobotEnv, ArmConstraintEnv
    print("✅ 环境模块导入成功")
except Exception as e:
    print(f"❌ 环境模块导入失败: {e}")
    sys.exit(1)

try:
    from src.training import TrainingConfig, Trainer, DebugCallback
    print("✅ 训练模块导入成功")
except Exception as e:
    print(f"❌ 训练模块导入失败: {e}")
    sys.exit(1)

try:
    config = EnvConfig.from_yaml("config/env/arm_constraint.yaml")
    print("✅ 配置文件加载成功")
except Exception as e:
    print(f"❌ 配置文件加载失败: {e}")
    sys.exit(1)

try:
    env = ArmConstraintEnv(config=config)
    print("✅ 环境创建成功")
    obs, info = env.reset()
    print(f"✅ 环境重置成功，观测维度: {obs.shape}")
    env.close()
except Exception as e:
    print(f"❌ 环境测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 所有测试通过！")

