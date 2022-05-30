import numpy as np
from typing import List, Tuple

from environment import Environment
from params import ist_stop_threshold, ist_tau
from utils import soft_threshold

def ist(env: Environment, target: Tuple[float, float], estimate: List[float], sanity_check=False) -> List[float]:
    """Runs centralized Iterative Soft Thresholding with an initial estimate x_0. Returns a 1-sparse vector"""
    B, z = env.measure_RSS(target)
    if sanity_check:
        if estimate == target:
            A, y = env.measure_RSS(target, use_feng=False)
            x_0 = estimate
            print("Does Ax = y hold?", np.array_equiv(A @ x_0, y))
            print("Does Bx = z (Feng orth.) hold?", np.allclose(B @ x_0, z, atol=1e-12, rtol=0)) # account for a small numerical error
        else:
            # print("Sanity checks skipped: estimate != target")
            pass
    x_t = estimate
    num_iterations = 0
    error_list = []
    x_t_list = []
    for _ in range(100000):
        num_iterations += 1
        prev_x_t = x_t
        x_t = soft_threshold(x_t + ist_tau * (B.T @ (z - B @ x_t).T))
        x_t_list.append(x_t)
        error = np.linalg.norm(x_t - prev_x_t, ord=2)
        error_list.append(error)
        if error <= ist_stop_threshold:
            break
    return x_t, num_iterations, error_list, x_t_list
