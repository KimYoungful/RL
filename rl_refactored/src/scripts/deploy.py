"""部署脚本：摄像头 -> 手部检测 -> 预测动作 -> 控制机器人（占位流程）。"""

import argparse
from pathlib import Path
import sys
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.env import EnvConfig
from src.deployment import RobotController, CameraStream, HandDetection, CameraCalibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--robot-config", required=False, default="config/deployment/robot.yaml")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    # 加载环境配置
    env_cfg = EnvConfig.from_yaml(args.env_config)

    # 读取机器人配置
    robot_ip = "192.168.1.2"
    try:
        import yaml
        with open(args.robot_config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        robot_ip = cfg.get("robot_ip", robot_ip)
    except Exception:
        pass

    controller = RobotController(
        model_path=args.model_path,
        env_config=env_cfg,
        robot_ip=robot_ip,
        algorithm="SAC",
    )

    camera = CameraStream(index=args.camera_index)
    detector = HandDetection()
    cali = CameraCalibration()

    try:
        for _ in range(1000):
            frame = camera.read()
            frame = cali.undistort_frame(frame)
            frame, hands = detector.process_frame(frame)
            # 简化：若未检测到手，跳过
            if not hands:
                continue
            hand_px = np.array(hands[0], dtype=np.float32)
            # 占位：将像素直接映射到环境坐标
            hand_env = hand_px / np.array([100.0, 100.0], dtype=np.float32)
            robot_env = np.array([5.0, 5.0], dtype=np.float32)  # 真实项目应使用实际机器人位置转换
            action = controller.predict_action(robot_env, hand_env)
            controller.step_robot(action, freq_hz=10.0)
    finally:
        controller.close()
        detector.release()
        camera.release()


if __name__ == "__main__":
    main()


