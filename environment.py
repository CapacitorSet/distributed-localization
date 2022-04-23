from typing import Tuple
from matplotlib import pyplot
from scipy.linalg import orth
from scipy.sparse.csgraph import connected_components
import itertools
import networkx as nx
import numpy as np
import numpy.typing as npt

from utils import build_dict_for_sensors, distance, RSS_model

class Environment:
    num_sensors: int
    reference_points: npt.ArrayLike # [0, 0], [0, 1].. [0, 9], [1, 0]... [9, 9]
    sensor_positions: npt.ArrayLike # eg. [[1.8 4.3], [5.5 7.6]]
    graph: npt.ArrayLike # connectivity graph: graph[i][j] = 1 if sensors i and j are connected

    _non_orthogonal_sensor_dict: npt.ArrayLike # the raw RSS dicts per sensor, before Feng orth.

    def __init__(self, num_sensors: int, connection_distance: float) -> None:
        self.num_sensors = num_sensors

        # Environment setup
        self.reference_points = np.array(list(itertools.product(
            np.linspace(0, 9, 10),
            np.linspace(0, 9, 10))))
        self.sensor_positions = 10*np.random.rand(num_sensors, 2)
        self.graph = np.zeros((num_sensors, num_sensors))
        for i in range(num_sensors):
            for j in range(num_sensors):
                self.graph[i][j] = i != j and distance(self.sensor_positions[i], self.sensor_positions[j]) <= connection_distance
        # nx.draw(nx.from_numpy_matrix(graph))
        # pyplot.show()
        assert self.is_connected(), "The connectivity graph is not connected"

        # Dict setup
        self._non_orthogonal_sensor_dict = build_dict_for_sensors(self.sensor_positions, self.reference_points)
    
    def is_connected(self) -> bool:
        """Checks if the sensor graph is connected, i.e. not partitioned"""
        num_components, _ = connected_components(self.graph, directed=False)
        # print("Connected components in sensor network:", num_components)
        return num_components == 1
    
    def measure_RSS(self, target, use_feng=True) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """Measures the RSS on each sensor for a given target. Returns the Feng orthogonalized matrices (B, z) or the raw (A, y)"""
        y = np.array([RSS_model(np.linalg.norm(target - sensor)) for sensor in self.sensor_positions])
        A = self._non_orthogonal_sensor_dict
        if not use_feng:
            return A, y
        # Feng orthogonalization
        B = orth(A.T).T
        A_dagger = np.linalg.pinv(A)
        z = B @ A_dagger @ y
        return B, z
