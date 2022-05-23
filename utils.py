import numpy as np
import itertools

from params import ist_lambda

def normalize(vec):
    return vec/np.linalg.norm(vec, ord=1)

def distance(a, b):
    return np.linalg.norm(a - b)

def soft_threshold_1d(x):
    """Applies soft thresholding to a number"""
    if abs(x) < ist_lambda:
        return 0.
    return x-np.sign(x)*ist_lambda

def soft_threshold(x):
    """Applies componentwise soft thresholding to a vector"""
    assert np.squeeze(x).ndim == 1, "soft_threshold takes a one-dimensional vector"
    return np.array([soft_threshold_1d(x_i) for x_i in np.asarray(x).reshape(-1)])

def show_1sparse_vector(x):
    x = normalize(x) # normalize x to get a vector of "probabilities"
    # these are not real probabilities but it is intuitive to present them as such
    entries = sorted([(val, *idx) for idx, val in np.ndenumerate(x)], reverse=True)
    # the indices refer to this array.
    reference_points = np.array(list(itertools.product(
            np.linspace(0, 9, 10),
            np.linspace(0, 9, 10))))
    print("Most likely values:")
    for prob, idx in entries[:5]:
        print(f" - {reference_points[idx]} (probability {100*prob:.3f}%)")