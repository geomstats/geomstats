"""
Plot the result of optimal quantization of the uniform distribution
on the circle.
"""

import matplotlib.pyplot as plt
import os

import geomstats.visualization as visualization

from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.quantization import Quantization

N_POINTS = 1000
N_CENTERS = 5
N_REPETITIONS = 20
TOLERANCE = 1e-6


def main():
    circle = Hypersphere(dimension=1)

    data = circle.random_uniform(n_samples=1000, bound=None)

    n_clusters = 5
    clustering = Quantization(metric=circle.metric, n_clusters=n_clusters)
    clustering = clustering.fit(data)

    #points = CIRCLE.random_uniform(n_samples=N_POINTS, bound=None)
    #centers, weights, clusters, n_iterations = METRIC.optimal_quantization(
    #            points=points, n_centers=N_CENTERS,
    #            n_repetitions=N_REPETITIONS, tolerance=TOLERANCE
    #            )

    plt.figure(0)
    visualization.plot(points=clustering.cluster_centers_, space='S1', color='red')
    plt.show()

    plt.figure(1)
    ax = plt.axes()
    circle_plot = visualization.Circle()
    circle_plot.draw(ax=ax)
    for i in range(n_clusters):
        cluster = data[clustering.labels_==i, :]
        circle_plot.draw_points(ax=ax, points=cluster)
    plt.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        print('Examples with visualizations are only implemented '
              'with numpy backend.\n'
              'To change backend, write: '
              'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
