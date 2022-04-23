from typing import List, Tuple
import numpy as np

from environment import Environment
from params import ist_max_iterations, ist_tau
from utils import soft_threshold

def local_dist_step(old_x, B, z, neighbors, sensor_id):
    """Performs one step of distributed IST on a single node"""
    consensus = np.average(old_x, axis=0, weights=neighbors)
    gradient = ist_tau * (B.T * (z - B @ old_x[sensor_id]).T)
    return (soft_threshold(consensus + gradient))

def dist(env: Environment, target: Tuple[float, float], estimate: List[float]) -> List[float]:
    """Performs distributed IST on a network. Returns a 1-sparse vector"""
    B, z = env.measure_RSS(target)
    x_t = estimate
    for _ in range(ist_max_iterations):
        x_t = np.array([local_dist_step(x_t, B[i], z[i], env.graph[i], i) for i in range(0, env.num_sensors)])
    return x_t