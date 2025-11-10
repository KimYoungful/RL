"""训练配置类"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml
import os


@dataclass
class TrainingConfig:
    """训练配置类"""
    
    algorithm: str = "SAC"  # SAC or PPO
    total_timesteps: int = 400000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    log_freq: int = 10000
    enable_debug_callback: bool = True
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TrainingConfig':
        """从YAML文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_yaml(self, yaml_path: str):
        """保存到YAML文件"""
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

