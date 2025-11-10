import numpy as np
from src.env import EnvConfig, BaseRobotEnv

def test_env_reset_and_step():
    cfg = EnvConfig()
    env = BaseRobotEnv(config=cfg)
    obs, info = env.reset()
    assert obs.shape[0] == env.observation_space.shape[0]
    a = env.action_space.sample()
    obs, rew, done, truncated, info = env.step(a)
    assert isinstance(rew, float)
    env.close()


