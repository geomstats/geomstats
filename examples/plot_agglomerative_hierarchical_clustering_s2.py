"""Plot an Agglomerative Hierarchical Clustering on the sphere."""

import logging
import os

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.agglomerative_hierarchical_clustering import (
    AgglomerativeHierarchicalClustering,
)


def main():
    """Plot an Agglomerative Hierarchical Clustering on the sphere."""
    sphere = Hypersphere(dim=2)

    n_clusters = 2
    n_samples_per_dataset = 50

    dataset_1 = sphere.random_von_mises_fisher(
        kappa=10, n_samples=n_samples_per_dataset
    )
    dataset_2 = -sphere.random_von_mises_fisher(
        kappa=10, n_samples=n_samples_per_dataset
    )
    dataset = gs.concatenate((dataset_1, dataset_2), axis=0)

    clustering = AgglomerativeHierarchicalClustering(sphere, n_clusters=n_clusters)

    clustering.fit(dataset)

    clustering_labels = clustering.labels_

    plt.figure(0)
    ax = plt.subplot(111, projection="3d")
    plt.title("Agglomerative Hierarchical Clustering")
    sphere_plot = visualization.Sphere()
    sphere_plot.draw(ax=ax)
    for i_label in range(n_clusters):
        points_label_i = dataset[clustering_labels == i_label, ...]
        sphere_plot.draw_points(ax=ax, points=points_label_i)

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
