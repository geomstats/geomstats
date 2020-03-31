"""
Plot the result of optimal quantization of the uniform distribution
on the circle, using online k-means clustering of a sample.
"""

import logging
import os

import matplotlib.pyplot as plt

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.online_kmeans import OnlineKMeans

N_POINTS = 1000
N_CENTERS = 5
N_REPETITIONS = 20
TOLERANCE = 1e-6


def main():
    circle = Hypersphere(dimension=1)

    data = circle.random_uniform(n_samples=1000)

    n_clusters = 5
    clustering = OnlineKMeans(metric=circle.metric, n_clusters=n_clusters)
    clustering = clustering.fit(data)

    plt.figure(0)
    visualization.plot(points=clustering.cluster_centers_, space='S1',
                       color='red')
    plt.show()

    plt.figure(1)
    ax = plt.axes()
    circle_plot = visualization.Circle()
    circle_plot.draw(ax=ax)
    for i in range(n_clusters):
        cluster = data[clustering.labels_ == i, :]
        circle_plot.draw_points(ax=ax, points=cluster)
    plt.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        logging.info('Examples with visualizations are only implemented '
                     'with numpy backend.\n'
                     'To change backend, write: '
                     'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
