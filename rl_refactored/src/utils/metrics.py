from typing import List
import numpy as np

def mean_distance(trajectory: List[np.ndarray]) -> float:
    if not trajectory:
        return 0.0
    distances = [t for t in trajectory if np.isscalar(t) or (hasattr(t, "shape") and t.shape == ())]
    if not distances:
        return 0.0
    return float(np.mean(distances))


