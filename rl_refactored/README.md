# 机器人避障强化学习项目（重构版）

## 项目简介

这是一个机器人避障的强化学习项目，使用Stable-Baselines3训练SAC/PPO算法，控制机器人在2D网格中避开手部障碍物。

## 项目结构

```
rl_refactored/
├── config/                  # 配置文件
│   ├── env/                # 环境配置
│   │   ├── base.yaml       # 基础环境配置
│   │   └── arm_constraint.yaml  # 带手臂约束的环境配置
│   └── training/           # 训练配置
│       └── sac.yaml        # SAC算法配置
├── src/                    # 源代码
│   ├── env/                # 环境模块
│   │   ├── config.py       # 环境配置类
│   │   ├── base_env.py     # 基础环境类
│   │   └── arm_env.py      # 带手臂约束的环境类
│   ├── training/           # 训练模块
│   │   ├── config.py       # 训练配置类
│   │   ├── trainer.py      # 训练器类
│   │   └── callbacks.py    # 回调函数
│   └── scripts/            # 脚本
│       └── train.py        # 训练脚本
├── logs/                   # 日志和模型（gitignore）
├── data/                   # 数据目录（gitignore）
└── tests/                  # 测试目录
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- gymnasium
- stable-baselines3
- numpy
- pygame
- pyyaml

## 使用方法

### 1. 训练模型

```bash
python src/scripts/train.py \
    --env-config config/env/arm_constraint.yaml \
    --training-config config/training/sac.yaml \
    --log-dir logs/training_run_1 \
    --env-type arm \
    --seed 42
```

### 2. 参数说明

- `--env-config`: 环境配置文件路径
- `--training-config`: 训练配置文件路径
- `--log-dir`: 日志和模型保存目录
- `--env-type`: 环境类型 (`arm` 或 `base`)
- `--seed`: 随机种子

### 3. 加载和使用模型

```python
from src.env import ArmConstraintEnv, EnvConfig
from stable_baselines3 import SAC

# 加载配置
config = EnvConfig.from_yaml("config/env/arm_constraint.yaml")
env = ArmConstraintEnv(config=config)

# 加载模型
model = SAC.load("logs/training_run_1/best_model/best_model.zip", env=env)

# 使用模型
obs, info = env.reset()
action, _ = model.predict(obs, deterministic=True)
obs, reward, done, truncated, info = env.step(action)
```

## 配置说明

### 环境配置

环境配置在 `config/env/` 目录下，主要包括：

- `grid_size`: 网格大小
- `margin`: 边界 margin
- `max_steps`: 最大步数
- `distance_threshold_*`: 距离阈值
- `reward_*`: 奖励参数
- `stride_*_range`: 步长范围
- `enable_domain_randomization`: 是否启用域随机化

### 训练配置

训练配置在 `config/training/` 目录下，主要包括：

- `algorithm`: 算法类型 (SAC/PPO)
- `total_timesteps`: 总训练步数
- `eval_freq`: 评估频率
- `model_kwargs`: 模型超参数

## 环境类型

### BaseRobotEnv

基础环境，机器人需要避开手部障碍物。

### ArmConstraintEnv

带手臂约束的环境，机器人不仅需要避开手部，还需要避开从手部到固定点的线段（手臂）。

## 重构改进

相比原项目，重构后的代码具有以下改进：

1. **配置管理**：所有参数集中在配置文件中，易于管理和实验
2. **模块化设计**：环境、训练、部署模块清晰分离
3. **类型提示**：添加了类型提示，提高代码可读性
4. **代码复用**：环境类可以在不同场景下复用
5. **易于扩展**：易于添加新功能和参数

## 开发计划

- [ ] 添加评估脚本
- [ ] 添加部署脚本
- [ ] 添加单元测试
- [ ] 添加可视化工具
- [ ] 完善文档

## 许可证

MIT License

## 作者

重构版本

