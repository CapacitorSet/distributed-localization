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

def norm_iter(x):
    val=np.linalg.norm(x,ord=2) 
    return val < dist_stop_threshold

def dist(env: Environment, target: Tuple[float, float], estimates: List[List[float]], max_iterations=1000, stubborn=True) -> List[float]:
    """Performs distributed IST on a network. Returns a 1-sparse vector"""

    if stubborn:
        idx_stubborn_sensor = np.random.randint(0,env.num_sensors)
        idx_stubborn_estimate = np.random.randint(0,100)
        estimates[idx_stubborn_sensor, :] = 0
        estimates[idx_stubborn_sensor, idx_stubborn_estimate] = 1

    B, z = env.measure_RSS(target)
    num_iterations = 0
    prev_estimates = estimates
    err_list = []
    for _ in range(max_iterations):
        num_iterations += np.apply_along_axis(lambda x: np.linalg.norm(x,ord=2) < dist_stop_threshold, 1, estimates - prev_estimates) 
        
        prev_estimates = estimates
        estimates = np.array([local_dist_step(estimates, B[i], z[i], env.graph[i], i) for i in range(0, env.num_sensors)])
        if stubborn:
            estimates[idx_stubborn_sensor, :] = 0
            estimates[idx_stubborn_sensor, idx_stubborn_estimate] = 1
        error = np.linalg.norm(estimates - prev_estimates, ord=2)
        err_list.append(error)
        if  error <= dist_stop_threshold:
            break
    return estimates, np.average(num_iterations), err_list