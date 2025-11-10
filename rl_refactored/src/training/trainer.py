"""训练器类"""

from pathlib import Path
from typing import Optional, Dict, Any
import yaml

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

from ..env import BaseRobotEnv, ArmConstraintEnv, EnvConfig
from .config import TrainingConfig
from .callbacks import DebugCallback


class Trainer:
    """训练器类"""
    
    def __init__(
        self,
        env_config: EnvConfig,
        training_config: TrainingConfig,
        log_dir: Path,
        env_class: type = ArmConstraintEnv
    ):
        self.env_config = env_config
        self.training_config = training_config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_class = env_class
        
        # 创建环境
        self.env = self._create_env()
        
        # 创建模型
        self.model = self._create_model()
        
    def _create_env(self):
        """创建环境"""
        def make_env():
            env = self.env_class(config=self.env_config)
            return Monitor(env)
        
        env = DummyVecEnv([make_env])
        env = VecMonitor(env)
        return env
    
    def _create_model(self):
        """创建模型"""
        model_class = SAC if self.training_config.algorithm == "SAC" else PPO
        
        # 默认模型参数
        default_kwargs = {
            "verbose": 1,
            "tensorboard_log": str(self.log_dir / "tensorboard"),
        }
        
        # 合并用户提供的参数
        model_kwargs = {**default_kwargs, **self.training_config.model_kwargs}
        
        model = model_class(
            "MlpPolicy",
            self.env,
            **model_kwargs
        )
        
        return model
    
    def train(self):
        """开始训练"""
        # 创建回调
        callbacks = self._create_callbacks()
        
        # 保存配置
        self._save_configs()
        
        # 开始训练
        print(f"开始训练，总步数: {self.training_config.total_timesteps}")
        self.model.learn(
            total_timesteps=self.training_config.total_timesteps,
            callback=callbacks
        )
        
        # 保存最终模型
        final_model_path = self.log_dir / "final_model.zip"
        self.model.save(str(final_model_path))
        print(f"模型已保存到: {final_model_path}")
    
    def _create_callbacks(self) -> CallbackList:
        """创建回调列表"""
        callbacks = []
        
        # 评估回调
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=str(self.log_dir / "best_model"),
            log_path=str(self.log_dir),
            eval_freq=self.training_config.eval_freq,
            deterministic=True,
            n_eval_episodes=self.training_config.n_eval_episodes,
        )
        callbacks.append(eval_callback)
        
        # 调试回调
        if self.training_config.enable_debug_callback:
            debug_callback = DebugCallback(
                env=self.env,
                log_freq=self.training_config.log_freq,
                verbose=1
            )
            callbacks.append(debug_callback)
        
        return CallbackList(callbacks)
    
    def _save_configs(self):
        """保存配置"""
        # 保存环境配置
        env_config_path = self.log_dir / "env_config.yaml"
        self.env_config.save_yaml(str(env_config_path))
        
        # 保存训练配置
        training_config_path = self.log_dir / "training_config.yaml"
        self.training_config.save_yaml(str(training_config_path))

