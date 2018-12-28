"""
Plot the result of optimal quantization of the uniform distribution
on the circle.
"""

import matplotlib.pyplot as plt

import geomstats.visualization as visualization

from geomstats.hypersphere import Hypersphere

CIRCLE = Hypersphere(dimension=1)
METRIC = CIRCLE.metric
N_POINTS = 1000
N_CENTERS = 5
N_REPETITIONS = 20
TOLERANCE = 1e-6


def main():
    points = CIRCLE.random_uniform(n_samples=N_POINTS, bound=None)

    centers, weights, clusters, n_iterations = METRIC.optimal_quantization(
                points=points, n_centers=N_CENTERS,
                n_repetitions=N_REPETITIONS, tolerance=TOLERANCE
                )

    plt.figure(0)
    visualization.plot(points=centers, space='S1', color='red')
    plt.show()

    plt.figure(1)
    ax = plt.axes()
    circle = visualization.Circle()
    circle.draw(ax=ax)
    for i in range(N_CENTERS):
        circle.draw_points(ax=ax, points=clusters[i])
    plt.show()


if __name__ == "__main__":
    main()
