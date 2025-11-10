from pathlib import Path
from src.env import EnvConfig
from src.training import TrainingConfig, Trainer

def test_trainer_init(tmp_path: Path):
    env_cfg = EnvConfig()
    tr_cfg = TrainingConfig(total_timesteps=1, enable_debug_callback=False)
    trainer = Trainer(env_cfg, tr_cfg, log_dir=tmp_path)
    assert trainer.env is not None
    assert trainer.model is not None


