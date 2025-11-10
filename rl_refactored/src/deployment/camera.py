"""摄像头采集与基础处理。"""

from typing import Optional, Tuple, List
import cv2


class CameraStream:
    """简单摄像头流包装。"""

    def __init__(self, index: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        self.cap = cv2.VideoCapture(index)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def release(self):
        if self.cap:
            self.cap.release()


