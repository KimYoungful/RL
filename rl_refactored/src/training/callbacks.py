"""训练回调函数"""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional


class DebugCallback(BaseCallback):
    """调试回调函数"""
    
    def __init__(
        self,
        env,
        render_freq: int = 10000,
        n_episodes: int = 1,
        log_freq: int = 10000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.termination_reasons = deque(maxlen=1000)
        self.distance_mean = deque(maxlen=1000)
        self.env_to_render = env
        self.render_freq = render_freq
        self.n_episodes = n_episodes
    
    def _on_step(self) -> bool:
        """每一步调用"""
        infos = self.locals.get('infos', None)
        dones = self.locals.get('dones', None)
        
        if infos is not None and dones is not None:
            for done, info in zip(dones, infos):
                if done and info is not None:
                    if 'done_reason' in info:
                        self.termination_reasons.append(info['done_reason'])
                    if 'distance_mean' in info:
                        self.distance_mean.append(info['distance_mean'])
        
        # 定期记录日志
        if self.num_timesteps % self.log_freq == 0 and self.verbose:
            self._log_metrics()
        
        return True
    
    def _log_metrics(self):
        """记录指标"""
        # 统计终止原因
        total = len(self.termination_reasons)
        if total > 0:
            count_out_of_bounds = sum(
                1 for r in self.termination_reasons
                if r == 'out_of_bounds'
            )
            ratio_out_of_bounds = count_out_of_bounds / total
        else:
            ratio_out_of_bounds = 0.0
        
        distance_mean = (
            sum(self.distance_mean) / len(self.distance_mean)
            if len(self.distance_mean) > 0 else 0.0
        )
        
        # 记录到TensorBoard
        self.logger.record("custom/termination_reason_ratio", ratio_out_of_bounds)
        self.logger.record("custom/distance_mean", distance_mean)
        self.logger.dump(step=self.num_timesteps)

