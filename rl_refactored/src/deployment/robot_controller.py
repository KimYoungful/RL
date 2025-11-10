"""机器人控制器：封装环境与模型推理，并对接机器人控制接口。"""

from typing import Optional
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC, PPO

from ..env import EnvConfig, BaseRobotEnv


class URControl:
    """UR 机器人控制占位实现（可替换为真实驱动）。"""

    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip

    def move_robot(self, target_pose, t_target: float):
        # 这里仅占位打印，实际项目中应调用 rtde_control 接口
        pass

    def get_robot_pose(self):
        # 返回占位位姿，真实项目应调用 rtde_receive
        return [0, 0, 0, 0, 0, 0]

    def disconnect(self):
        pass


class RobotController:
    """加载模型，使用环境构造观测，输出动作，并控制机器人。"""

    def __init__(
        self,
        model_path: str,
        env_config: EnvConfig,
        robot_ip: str,
        algorithm: str = "SAC",
    ):
        self.env = BaseRobotEnv(config=env_config, render_mode=None)
        self.env.random = False

        algo = algorithm.upper()
        if algo == "SAC":
            self.model = SAC.load(
                model_path,
                env=self.env,
                custom_objects={
                    "observation_space": self.env.observation_space,
                    "action_space": self.env.action_space,
                },
            )
        else:
            self.model = PPO.load(
                model_path,
                env=self.env,
                custom_objects={
                    "observation_space": self.env.observation_space,
                    "action_space": self.env.action_space,
                },
            )

        self.robot = URControl(robot_ip)
        self.last_action = np.zeros(2, dtype=np.float32)

    def predict_action(
        self,
        robot_position: np.ndarray,
        hand_position: np.ndarray,
    ) -> np.ndarray:
        distance = np.linalg.norm(robot_position - hand_position)
        boundary = min(
            robot_position[0],
            robot_position[1],
            self.env.config.grid_size * 2 - robot_position[0],
            self.env.config.grid_size - robot_position[1],
        )
        obs = np.concatenate(
            [
                robot_position,
                hand_position,
                self.last_action,
                np.array([distance], dtype=np.float32),
                np.array([boundary], dtype=np.float32),
            ]
        ).astype(np.float32)

        action, _ = self.model.predict(obs, deterministic=True)
        self.last_action = action.astype(np.float32)
        return action

    def step_robot(self, action: np.ndarray, freq_hz: float = 10.0):
        # 将动作转成机器人增量运动（占位）
        # 真实项目中应完成像素/环境系 → 机械臂工作空间坐标的变换与安全限幅
        pose = self.robot.get_robot_pose()
        t = 1.0 / max(freq_hz, 1e-6)
        self.robot.move_robot(pose, t)

    def close(self):
        self.robot.disconnect()
        self.env.close()


