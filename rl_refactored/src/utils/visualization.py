from typing import List, Tuple
import pygame

def draw_trajectory(surface, points: List[Tuple[int, int]], color=(0, 0, 255), width=2):
    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, width)
        for p in points:
            pygame.draw.circle(surface, color, p, 3)


