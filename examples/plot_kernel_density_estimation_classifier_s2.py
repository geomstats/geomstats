"""Plot a Kernel Density Estimation Classification on the sphere."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kernel_density_estimation_classifier import (
    KernelDensityEstimationClassifier,
)
from geomstats.learning.radial_kernel_functions import triangular_radial_kernel


def main():
    """Plot a Kernel Density Estimation Classification on the sphere."""
    sphere = Hypersphere(dim=2)

    n_labels = 2
    n_samples_per_dataset = 10
    n_targets = 200
    radius = np.inf

    kernel = triangular_radial_kernel
    bandwidth = 3

    n_training_samples = n_labels * n_samples_per_dataset
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

    labels_colors = gs.zeros([n_labels, 3])
    labels_colors[0, :] = gs.array([0, 0, 1])
    labels_colors[1, :] = gs.array([1, 0, 0])

    kde = KernelDensityEstimationClassifier(
        sphere,
        radius=radius,
        kernel=kernel,
        bandwidth=bandwidth,
        outlier_label="most_frequent",
    )
    kde.fit(training_dataset, labels)
    target_labels = kde.predict(target)
    target_labels_proba = kde.predict_proba(target)

    plt.figure(0)
    ax = plt.subplot(111, projection="3d")
    plt.title("Training set")
    sphere_plot = visualization.Sphere()
    sphere_plot.draw(ax=ax)
    colors = gs.zeros([n_training_samples, 3])
    for i_sample in range(n_training_samples):
        colors[i_sample, :] = labels_colors[labels[i_sample], :]
    sphere_plot.draw_points(ax=ax, points=training_dataset, c=colors)

    plt.figure(1)
    ax = plt.subplot(111, projection="3d")
    plt.title("Classification")
    sphere_plot = visualization.Sphere()
    sphere_plot.draw(ax=ax)
    colors = gs.zeros([n_targets, 3])
    for i_target in range(n_targets):
        colors[i_target, :] = labels_colors[target_labels[i_target], :]
    sphere_plot.draw_points(ax=ax, points=target, c=colors)

    plt.figure(2)
    ax = plt.subplot(111, projection="3d")
    plt.title("Probabilistic classification")
    sphere_plot = visualization.Sphere()
    sphere_plot.draw(ax=ax)
    colors = target_labels_proba @ labels_colors
    sphere_plot.draw_points(ax=ax, points=target, c=colors)

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
