import numpy as np
from matplotlib import pyplot
import matplotlib
matplotlib.use('TkAgg')

from environment import Environment
from dist import dist
from ist import ist
from params import *
from utils import *

env = Environment(num_sensors=num_sensors, connection_distance=connection_distance)

target = env.reference_points[0]
# Estimate that the target is in position 3
ist_estimate = np.zeros(100)
ist_estimate[3] = 1
x = ist(env, target, ist_estimate, sanity_check=True)
show_1sparse_vector(x)
# pyplot.plot(x)
# pyplot.show()


# Estimates _for each sensor_
# dist_estimate = np.random.random((25, 100)) * 1e-2
dist_estimate = np.zeros((25, 100))
for i in range(0, num_sensors):
    # Initial estimate: the target is in position 3
    dist_estimate[i][3] = 1
x = dist(env, target, dist_estimate)
x = x[0] # Inspect data for first sensor
show_1sparse_vector(x)
pyplot.plot(x)
pyplot.show()
