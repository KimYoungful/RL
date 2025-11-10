# 快速开始指南

## 1. 安装依赖

```bash
cd rl_refactored
pip install -r requirements.txt
```

## 2. 测试环境

运行示例代码测试环境是否正常工作：

```bash
python example_usage.py
```

## 3. 训练模型

### 方式1: 使用训练脚本

```bash
python src/scripts/train.py \
    --env-config config/env/arm_constraint.yaml \
    --training-config config/training/sac.yaml \
    --log-dir logs/my_training \
    --env-type arm \
    --seed 42
```

### 方式2: 使用Python代码

```python
from src.env import ArmConstraintEnv, EnvConfig
from src.training import TrainingConfig, Trainer
from pathlib import Path

# 加载配置
env_config = EnvConfig.from_yaml("config/env/arm_constraint.yaml")
training_config = TrainingConfig.from_yaml("config/training/sac.yaml")

# 创建训练器
trainer = Trainer(
    env_config=env_config,
    training_config=training_config,
    log_dir=Path("logs/my_training"),
    env_class=ArmConstraintEnv
)

# 开始训练
trainer.train()
```

## 4. 加载和使用模型

```python
from src.env import ArmConstraintEnv, EnvConfig
from stable_baselines3 import SAC

# 加载配置和环境
config = EnvConfig.from_yaml("config/env/arm_constraint.yaml")
env = ArmConstraintEnv(config=config)

# 加载模型
model = SAC.load("logs/my_training/best_model/best_model.zip", env=env)

# 运行一个episode
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()  # 可选：渲染环境

print(f"总奖励: {total_reward}")
env.close()
```

## 5. 修改配置

### 修改环境参数

编辑 `config/env/arm_constraint.yaml`:

```yaml
grid_size: 10.0
max_steps: 50
distance_threshold_collision: 1.5
# ... 其他参数
```

### 修改训练参数

编辑 `config/training/sac.yaml`:

```yaml
total_timesteps: 400000
eval_freq: 10000
model_kwargs:
  learning_rate: 0.0003
  # ... 其他参数
```

## 6. 查看训练日志

训练日志会保存在 `logs/` 目录下：

- `logs/my_training/best_model/`: 最佳模型
- `logs/my_training/tensorboard/`: TensorBoard日志
- `logs/my_training/env_config.yaml`: 环境配置备份
- `logs/my_training/training_config.yaml`: 训练配置备份

使用TensorBoard查看训练过程：

```bash
tensorboard --logdir logs/my_training/tensorboard
```

## 7. 常见问题

### Q: 导入错误

确保在项目根目录运行脚本，或者将项目根目录添加到PYTHONPATH：

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Q: 找不到hand.png

确保 `hand.png` 文件在项目根目录或正确配置路径。

### Q: 训练速度慢

可以：
1. 减少 `total_timesteps`
2. 减少网络大小
3. 使用GPU加速（如果可用）

## 8. 下一步

- 查看 `README.md` 了解更多信息
- 查看 `重构建议.md` 了解重构详情
- 修改配置进行实验
- 添加自定义回调函数
- 实现评估脚本

