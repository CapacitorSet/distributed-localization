import numpy as np
from typing import List, Tuple

from environment import Environment
from params import ist_max_iterations, ist_tau
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
            print("Sanity checks skipped: estimate != target")
    x_t = estimate
    for _ in range(ist_max_iterations):
        x_t = soft_threshold(x_t + ist_tau * (B.T @ (z - B @ x_t).T))
    return x_t
