import numpy as np
from scipy.linalg import orth
from scipy.sparse.csgraph import connected_components
import networkx as nx
from matplotlib import pyplot
import matplotlib
matplotlib.use('TkAgg')
import itertools

num_sensors = 25
num_measurements = 1
RSS_std_dev = 0 # .5 # standard deviation of the RSS noise
connection_distance = 4 # the threshold distance for two sensors to be connected
ist_lambda = 1e-4
ist_tau = .7
ist_max_iterations = 1000
np.random.seed(1)

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

reference_points = np.array(list(itertools.product(np.linspace(0, 9, 10), np.linspace(0, 9, 10)))) # [0, 0], [0, 1].. [0, 9], [1, 0]... [9, 9]
sensor_positions = 10*np.random.rand(num_sensors, 2) # eg. [[1.8 4.3], [5.5 7.6]]
graph = np.zeros((num_sensors, num_sensors))
for i in range(num_sensors):
    for j in range(num_sensors):
        graph[i][j] = distance(sensor_positions[i], sensor_positions[j]) <= connection_distance
        # graph[i][j] = i != j and distance(sensor_positions[i], sensor_positions[j]) <= connection_distance
# nx.draw(nx.from_numpy_matrix(graph))
# pyplot.show()
num_components, _ = connected_components(graph, directed=False)
print("Connected components in sensor network:", num_components)
assert num_components == 1, "The connectivity graph is not connected"

def build_dict_for_sensor(sensor):
    return [RSS_model(distance(point, sensor)) for point in reference_points]
A = np.array([build_dict_for_sensor(sensor) for sensor in sensor_positions])

def feng_orthogonalization(A, y):
    B = orth(A.T).T
    A_dagger = np.linalg.pinv(A)
    z = B @ A_dagger @ y
    return B, z

target = reference_points[0] # todo: randomize
y = np.array([RSS_model(np.linalg.norm(target - sensor)) for sensor in sensor_positions])
B, z = feng_orthogonalization(A, y)

def soft_threshold_1d(x):
    """Applies soft thresholding to a number"""
    if abs(x) < ist_lambda:
        return 0.
    return x-np.sign(x)*ist_lambda

def soft_threshold(x):
    """Applies componentwise soft thresholding to a vector"""
    assert np.squeeze(x).ndim == 1, "soft_threshold takes a one-dimensional vector"
    return np.array([soft_threshold_1d(x_i) for x_i in np.asarray(x).reshape(-1)])

def ist(x_0, A, y):
    """Implements centralized Iterative Soft Thresholding with an initial estimate x_0"""
    x_t = x_0
    for _ in range(ist_max_iterations):
        x_t = soft_threshold(x_t + ist_tau * (A.T @ (y - A @ x_t).T))
    return x_t

x_0 = np.zeros(100)
x_0[3] = 1
# Sanity checks. These only apply if x_0 equals the target position
# print("Does Ax = y hold?", np.array_equiv(A @ x_0, y))
# print("Does Bx = z (Feng orth.) hold?", np.allclose(B @ x_0, z, atol=1e-12, rtol=0)) # account for a small numerical error
x = ist(x_0, B, z)
# pyplot.plot(x)
# pyplot.show()

x = x/np.linalg.norm(x, ord=1) # normalize x to get a vector of "probabilities"
# these are not real probabilities but it is intuitive to present them as such
entries = sorted([(val, *idx) for idx, val in np.ndenumerate(x)], reverse=True)
print("Most likely values:")
for prob, idx in entries[:5]:
    print(f" - #{idx} (probability {100*prob:.3f}%)")

def local_dist_step(old_x, sensor_id):
    """Performs one step of distributed IST on a single node"""
    local_A = B[sensor_id] # should have shape (num_targets, )
    local_y = z[sensor_id]
    somma = sum([graph[sensor_id][j] * old_x[j] for j in range(0, num_sensors)])
    gradiente = ist_tau * (local_A.T * (local_y - local_A @ old_x[sensor_id]).T)
    return soft_threshold(somma + gradiente)

def local_dist(x_0):
    """Performs distributed IST on a network"""
    x_t = x_0
    for _ in range(ist_max_iterations): # for t in 0..Tmax
        x_t = np.array([local_dist_step(x_t, i) for i in range(0, num_sensors)])
    return x_t

x_0 = np.random.random((25, 100)) * 1e-2 # Estimates _for each sensor_
# for i in range(0, num_sensors):
    # Initial estimate: the target is in position 3
    # x_0[i][3] = 1
ist_max_iterations = 4
ist_tau = .2
x = local_dist(x_0)

x = x[0]
pyplot.plot(x)
pyplot.show()
x = x/np.linalg.norm(x, ord=1) # normalize x to get a vector of "probabilities"
# these are not real probabilities but it is intuitive to present them as such
entries = sorted([(val, *idx) for idx, val in np.ndenumerate(x)], reverse=True)
print("Most likely values:")
for prob, idx in entries[:5]:
    print(f" - #{idx} (probability {100*prob:.3f}%)")