"""相机标定占位（接口兼容）。"""

import numpy as np


class CameraCalibration:
    def __init__(self):
        pass

    def undistort_frame(self, frame):
        # 占位：直接返回
        return frame

    def world_to_pixel(self, world_xy):
        # 占位：1:1 映射
        x, y = world_xy[:2]
        return np.array([x, y], dtype=np.float32)

    def pixel_to_world(self, pixel_xy):
        x, y = pixel_xy[:2]
        return np.array([x, y], dtype=np.float32)


