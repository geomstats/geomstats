"""Plot the result of online K-means, i.e. optimal quantization.

This is online K-means of the von Mises Fisher distribution
on the sphere using online k-means clustering of a sample.
"""

import logging
import os

import matplotlib.pyplot as plt

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.online_kmeans import OnlineKMeans


def main():
    """Run online K-means on the sphere."""
    sphere = Hypersphere(dim=2)

    data = sphere.random_von_mises_fisher(kappa=10, n_samples=1000)

    n_clusters = 4
    clustering = OnlineKMeans(sphere.metric, n_clusters=n_clusters)
    clustering = clustering.fit(data)

    plt.figure(0)
    ax = plt.subplot(111, projection="3d")
    visualization.plot(points=clustering.cluster_centers_, ax=ax, space="S2", c="r")
    plt.show()

    plt.figure(1)
    ax = plt.subplot(111, projection="3d")
    sphere_plot = visualization.Sphere()
    sphere_plot.draw(ax=ax)
    for i in range(n_clusters):
        cluster = data[clustering.labels_ == i, :]
        sphere_plot.draw_points(ax=ax, points=cluster)
    plt.show()


if __name__ == "__main__":
    if os.environ.get("GEOMSTATS_BACKEND", "numpy") != "numpy":
        logging.info(
            "Examples with visualizations are only implemented "
            "with numpy backend.\n"
            "To change backend, write: "
            "export GEOMSTATS_BACKEND = 'numpy'."
        )
    else:
        main()
