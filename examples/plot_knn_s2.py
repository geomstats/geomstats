"""Plot the result of a KNN classification on the sphere."""

import logging
import os

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.knn import KNearestNeighborsClassifier


def main():
    """Plot the result of a KNN classification on the sphere."""
    sphere = Hypersphere(dim=2)
    sphere_distance = sphere.metric.dist

    n_labels = 2
    n_samples_per_dataset = 10
    n_targets = 200

    dataset_1 = sphere.random_von_mises_fisher(
        kappa=10, n_samples=n_samples_per_dataset
    )
    dataset_2 = -sphere.random_von_mises_fisher(
        kappa=10, n_samples=n_samples_per_dataset
    )
    training_dataset = gs.concatenate((dataset_1, dataset_2), axis=0)
    labels_dataset_1 = gs.zeros([n_samples_per_dataset], dtype=gs.int64)
    labels_dataset_2 = gs.ones([n_samples_per_dataset], dtype=gs.int64)
    labels = gs.concatenate((labels_dataset_1, labels_dataset_2))
    target = sphere.random_uniform(n_samples=n_targets)

    neigh = KNearestNeighborsClassifier(n_neighbors=2, distance=sphere_distance)
    neigh.fit(training_dataset, labels)
    target_labels = neigh.predict(target)

    plt.figure(0)
    ax = plt.subplot(111, projection="3d")
    plt.title("Training set")
    sphere_plot = visualization.Sphere()
    sphere_plot.draw(ax=ax)
    for i_label in range(n_labels):
        points_label_i = training_dataset[labels == i_label, ...]
        sphere_plot.draw_points(ax=ax, points=points_label_i)

    plt.figure(1)
    ax = plt.subplot(111, projection="3d")
    plt.title("Classification")
    sphere_plot = visualization.Sphere()
    sphere_plot.draw(ax=ax)
    for i_label in range(n_labels):
        target_points_label_i = target[target_labels == i_label, ...]
        sphere_plot.draw_points(ax=ax, points=target_points_label_i)

    plt.show()


if __name__ == "__main__":
    if os.environ["GEOMSTATS_BACKEND"] != "numpy":
        logging.info(
            "Examples with visualizations are only implemented "
            "with numpy backend.\n"
            "To change backend, write: "
            "export GEOMSTATS_BACKEND = 'numpy'."
        )
    else:
        main()
