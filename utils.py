import numpy as np
from scipy.linalg import orth

from params import RSS_std_dev, ist_lambda

def normalize(vec):
    return vec/np.linalg.norm(vec, ord=1)

def RSS_model(d):
    """Returns the received signal strength according to IEEE 802.15.4."""
    P_t = 25
    eta = np.random.normal(scale=RSS_std_dev) # noise
    if d <= 8:
        return P_t - 40.2 - 20*np.log(d) + eta
    else:
        return P_t - 58.5 - 33*np.log(d) + eta
def distance(a, b):
    return np.linalg.norm(a - b)

def build_dict_for_sensor(sensor, reference_points):
    return [RSS_model(distance(point, sensor)) for point in reference_points]
def build_dict_for_sensors(sensor_positions, reference_points):
    return np.array([build_dict_for_sensor(sensor, reference_points) for sensor in sensor_positions])

def feng_orthogonalization(A, y):
    B = orth(A.T).T
    A_dagger = np.linalg.pinv(A)
    z = B @ A_dagger @ y
    return B, z

def soft_threshold_1d(x):
    """Applies soft thresholding to a number"""
    if abs(x) < ist_lambda:
        return 0.
    return x-np.sign(x)*ist_lambda

def soft_threshold(x):
    """Applies componentwise soft thresholding to a vector"""
    assert np.squeeze(x).ndim == 1, "soft_threshold takes a one-dimensional vector"
    return np.array([soft_threshold_1d(x_i) for x_i in np.asarray(x).reshape(-1)])