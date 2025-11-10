"""手部检测（基于Mediapipe）。"""

from typing import List, Tuple
import mediapipe as mp
import cv2
import numpy as np


class HandDetection:
    def __init__(self, max_num_hands: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        positions = []
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for lm in results.multi_hand_landmarks:
                pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))
                positions.append((cx, cy))
                self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)
        return frame, positions

    def release(self):
        self.hands.close()


