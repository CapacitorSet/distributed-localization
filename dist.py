from typing import List, Tuple
import numpy as np

from environment import Environment
from params import ist_tau, dist_stop_threshold
from utils import soft_threshold

def local_dist_step(old_x, B, z, neighbors, sensor_id):
    """Performs one step of distributed IST on a single node"""
    consensus = np.average(old_x, axis=0, weights=neighbors)
    gradient = ist_tau * (B.T * (z - B @ old_x[sensor_id]).T)
    # Todo: gradient computation could be vectorized (eg. vectorized B[i] @ x_t[i] is np.sum(B*x_t, axis=1))
    return (soft_threshold(consensus + gradient))

def dist(env: Environment, target: Tuple[float, float], estimates: List[List[float]], max_iterations=1000) -> List[float]:
    """Performs distributed IST on a network. Returns a 1-sparse vector"""

    B, z = env.measure_RSS(target)
    num_iterations = 0
    for _ in range(max_iterations):
        num_iterations += 1
        prev_estimates = estimates
        estimates = np.array([local_dist_step(estimates, B[i], z[i], env.graph[i], i) for i in range(0, env.num_sensors)])
        if np.linalg.norm(estimates - prev_estimates, ord=2) <= dist_stop_threshold:
            break
    return estimates, num_iterations