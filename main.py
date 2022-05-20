import multiprocessing
from argparse import ArgumentParser

import matplotlib
import numpy as np
matplotlib.use('TkAgg')

from dist import dist
from environment import Environment
from ist import ist
from params import *
from utils import *


def run_simulation(seed):
    try:
        np.random.seed(seed)
        env = Environment(num_sensors=num_sensors, connection_distance=connection_distance)

        target_idx = np.random.randint(0, 100) # Pick a random target every time, except for O-DIST
        print(f"Running simulation with seed={seed}, target={env.reference_points[target_idx]}")

        with open("ist.csv", "a", buffering=1) as results_file:
            # print("IST")
            target = env.reference_points[target_idx]
            # Start with a random estimation
            initial_estimate_idx = np.random.randint(0, 100)
            initial_estimate = np.zeros(100)
            initial_estimate[initial_estimate_idx] = 1
            # print("Initial estimate:", env.reference_points[initial_estimate_idx])
            ist_estimate, num_iterations = ist(env, target, initial_estimate, sanity_check=True)
            # position estimate = the ref. point corresponding to the largest component in the est. vector
            position_estimate = env.reference_points[np.argmax(ist_estimate)]
            # print("IST estimate:", position_estimate)
            error = np.linalg.norm(position_estimate - target, ord=2)
            results_file.write(f"{seed};{error:.3f};{num_iterations}\n")
            # show_1sparse_vector(ist_estimate)
            # pyplot.plot(x)
            # pyplot.show()
            # print()

        with open("dist.csv", "a", buffering=1) as results_file:
            # print("DIST")
            # Estimates _for each sensor_
            initial_estimate = np.zeros((25, 100))
            for i in range(0, num_sensors):
                # Initial estimate: the target is in position 3
                initial_estimate[i][3] = 1
            dist_estimate, num_iterations = dist(env, target, initial_estimate, max_iterations=100000)
            average_estimate = np.average(dist_estimate, axis=0)
            position_estimate = env.reference_points[np.argmax(average_estimate)]
            # print("DIST estimate:", position_estimate)
            error = np.linalg.norm(position_estimate - target, ord=2)
            results_file.write(f"{seed};{error:.3f};{num_iterations};{env.essential_spectral_radius():.3f}\n")
            # show_1sparse_vector(average_estimate)
            # pyplot.plot(x)
            # pyplot.show()
            # print()

        with open("o-dist.csv", "a", buffering=1) as results_file:
            # O-DIST
            dist_estimate = np.zeros((25, 100))
            for i in range(0, num_sensors):
                # Initial estimate: the target is in position 0
                dist_estimate[i][0] = 1
            target = [0, 0]
            movement_direction = "x" # alternates between x and y at each iteration
            i = 0
            cumulative_error = 0
            while target[0] != 10 and target[1] != 10:
                # print(f"O-DIST iteration #{i}, target is {target}")
                # dist_estimate = dist(env, target, dist_estimate, max_iterations=500)
                dist_estimate, _ = dist(env, target, dist_estimate, max_iterations=500)
                average_estimate = np.average(dist_estimate, axis=0)
                # show_1sparse_vector(average_estimate)
                position_estimate = env.reference_points[np.argmax(average_estimate)]
                error = np.linalg.norm(position_estimate - target, ord=2)
                cumulative_error += error
                # print(f"Position estimate: {position_estimate}")
                # print(f"Error: {error}")
                # print(f"Cumulative error: {cumulative_error}")
                # print()
                results_file.write(f"{seed};{i};{error:.3f};{cumulative_error:.3f}\n")

                if movement_direction == "x":
                    target[0] = min(target[0] + 1, 10)
                    movement_direction = "y"
                else:
                    target[1] = min(target[1] + 1, 10)
                    movement_direction = "x"
                i = i + 1
    except Exception as e:
        # Do not crash everything if one job crashes
        print(e)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-j", "--jobs", dest="jobs", metavar="JOBS", type=int,
                        help="How many simulation jobs to run (default: one per CPU thread)",
                        default=multiprocessing.cpu_count())
    parser.add_argument("-r", "--runs", dest="runs", metavar="RUNS", default=50, type=int,
                        help="How many simulations/seeds to run (default: 50)")
    args = parser.parse_args()
    print(args)

    with multiprocessing.Pool(args.jobs) as p:
        i = 0
        for _ in p.imap_unordered(run_simulation, range(0, args.runs)):
            i = i + 1
            print(f"Simulation progress: {100*i/args.runs}%")