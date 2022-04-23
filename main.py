import numpy as np
from scipy.sparse.csgraph import connected_components
import networkx as nx
from matplotlib import pyplot
import matplotlib
matplotlib.use('TkAgg')
import itertools

from params import *
from utils import *

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

target = reference_points[0] # todo: randomize
A = build_dict_for_sensors(sensor_positions, reference_points)
y = np.array([RSS_model(np.linalg.norm(target - sensor)) for sensor in sensor_positions])
B, z = feng_orthogonalization(A, y)

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

x = normalize(x) # normalize x to get a vector of "probabilities"
# these are not real probabilities but it is intuitive to present them as such
entries = sorted([(val, *idx) for idx, val in np.ndenumerate(x)], reverse=True)
print("Most likely values:")
for prob, idx in entries[:5]:
    print(f" - #{idx} (probability {100*prob:.3f}%)")

def local_dist_step(old_x, sensor_id):
    """Performs one step of distributed IST on a single node"""
    local_A = B[sensor_id] # should have shape (num_targets, )
    local_y = z[sensor_id]
    consensus = sum([graph[sensor_id][j] * old_x[j] for j in range(0, num_sensors)])
    gradient = ist_tau * (local_A.T * (local_y - local_A @ old_x[sensor_id]).T)
    return (soft_threshold(consensus + gradient))

def local_dist(x_0):
    """Performs distributed IST on a network"""
    x_t = x_0
    for _ in range(ist_max_iterations): # for t in 0..Tmax
        x_t = np.array([local_dist_step(x_t, i) for i in range(0, num_sensors)])
    return x_t

# Estimates _for each sensor_
# x_0 = np.random.random((25, 100)) * 1e-2
x_0 = np.zeros((25, 100))
for i in range(0, num_sensors):
    # Initial estimate: the target is in position 3
    x_0[i][3] = 1
x = local_dist(x_0)

x = x[0]
pyplot.plot(x)
pyplot.show()
x = normalize(x) # normalize x to get a vector of "probabilities"
# these are not real probabilities but it is intuitive to present them as such
entries = sorted([(val, *idx) for idx, val in np.ndenumerate(x)], reverse=True)
print("Most likely values:")
for prob, idx in entries[:5]:
    print(f" - #{idx} (probability {100*prob:.3f}%)")