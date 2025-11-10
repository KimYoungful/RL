"""训练模块"""

from .config import TrainingConfig
from .trainer import Trainer
from .callbacks import DebugCallback

__all__ = ["TrainingConfig", "Trainer", "DebugCallback"]

