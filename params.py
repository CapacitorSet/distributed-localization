import numpy as np

num_sensors = 25
num_measurements = 1
RSS_std_dev = 0 # .5 # standard deviation of the RSS noise
connection_distance = 4 # the threshold distance for two sensors to be connected
ist_lambda = 1e-4
ist_tau = .7
ist_max_iterations = 1000
np.random.seed(1)