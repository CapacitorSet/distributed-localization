from typing import Tuple
from scipy.linalg import orth
from scipy.sparse.csgraph import connected_components
import itertools
import networkx as nx
import numpy as np
import numpy.typing as npt

from utils import distance

class Environment:
    num_sensors: int
    RSS_std_dev: float
    reference_points: npt.ArrayLike # [0, 0], [0, 1].. [0, 9], [1, 0]... [9, 9]
    sensor_positions: npt.ArrayLike # eg. [[1.8 4.3], [5.5 7.6]]
    graph: npt.ArrayLike # connectivity graph: graph[i][j] = 1 if sensors i and j are connected
    plot: bool # True if you want to plot the error at each iteration for a simulation (not suggested for many simulations)
    stubborn: bool
    _non_orthogonal_sensor_dict: npt.ArrayLike # the raw RSS dicts per sensor, before Feng orth.

    def __init__(self, num_sensors: int, connection_distance: float, RSS_std_dev: float, seed: int = None, stubborn: int=0, plot: bool = False) -> None:
        if seed is not None:
            np.random.seed(seed)
        self.plot = plot
        self.num_sensors = num_sensors
        self.RSS_std_dev = RSS_std_dev
        # Environment setup
        self.reference_points = np.array(list(itertools.product(
            np.linspace(0, 9, 10),
            np.linspace(0, 9, 10))))
        self.sensor_positions = 10*np.random.rand(num_sensors, 2)
        self.graph = np.zeros((num_sensors, num_sensors))
        for i in range(num_sensors):
            for j in range(num_sensors):
                self.graph[i][j] = i != j and distance(self.sensor_positions[i], self.sensor_positions[j]) <= connection_distance
        self.graph = self.initialize_Q(self.graph)
        # nx.draw(nx.from_numpy_matrix(graph))
        # pyplot.show()
        assert self.is_connected(), "The connectivity graph is not connected"

        # Dict setup
        self._non_orthogonal_sensor_dict = np.array([self.build_dict_for_sensor(sensor) for sensor in self.sensor_positions])

        self.csv_header = f"{seed};{num_sensors};{connection_distance};{RSS_std_dev};{stubborn}"
        print(self.graph.astype(float))
        #print(np.all(np.transpose(self.graph) == self.graph))
    

    def initialize_Q(self, Q_ones: npt.ArrayLike) -> npt.ArrayLike:
        Q = Q_ones

        for i in range(len(Q_ones)):
            in_degree_den = np.sum(Q_ones[i,:])+1
            Q[i,:] /= in_degree_den
        
        return Q

    def is_connected(self) -> bool:
        """Checks if the sensor graph is connected, i.e. not partitioned"""
        num_components, _ = connected_components(self.graph, directed=False)
        #print("Connected components in sensor network:", num_components)
        return num_components == 1
    
    def essential_spectral_radius(self) -> int:
        "Returns the 2nd largest eigenvalue of the sensor graph"
        #print(nx.adjacency_spectrum(nx.from_numpy_array(np.array(self.graph))).real)
        eig_val, eig_vec = np.linalg.eig(self.graph)

        #reverse the sorted vector so that now it's in descending order
        eig_val = sorted(eig_val.real)[::-1] 
        return eig_val[1]
        
    def RSS_model(self, d):
        """Returns the received signal strength according to IEEE 802.15.4."""
        P_t = 25
        eta = np.random.normal(scale=self.RSS_std_dev) # noise
        if d <= 8:
            return P_t - 40.2 - 20*np.log(d) + eta
        else:
            return P_t - 58.5 - 33*np.log(d) + eta

    def build_dict_for_sensor(self, sensor):
        return [self.RSS_model(distance(point, sensor)) for point in self.reference_points]

    def measure_RSS(self, target, use_feng=True) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """Measures the RSS on each sensor for a given target. Returns the Feng orthogonalized matrices (B, z) or the raw (A, y)"""
        y = np.array([self.RSS_model(distance(target, sensor)) for sensor in self.sensor_positions])
        A = self._non_orthogonal_sensor_dict
        if not use_feng:
            return A, y
        # Feng orthogonalization
        B = orth(A.T).T
        A_dagger = np.linalg.pinv(A)
        z = B @ A_dagger @ y
        return B, z
