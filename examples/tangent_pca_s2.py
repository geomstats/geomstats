"""Perform tangent PCA at the mean on the sphere."""

import logging

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA


def main():
    """Perform tangent PCA at the mean on the sphere."""
    fig = plt.figure(figsize=(15, 5))

    sphere = Hypersphere(dim=2)

    data = sphere.random_von_mises_fisher(kappa=15, n_samples=140)

    mean = FrechetMean(sphere)
    mean.fit(data)

    mean_estimate = mean.estimate_

    tpca = TangentPCA(sphere, n_components=2)
    tpca = tpca.fit(data, base_point=mean_estimate)
    tangent_projected_data = tpca.transform(data)

    geodesic_0 = sphere.metric.geodesic(
        initial_point=mean_estimate, initial_tangent_vec=tpca.components_[0]
    )
    geodesic_1 = sphere.metric.geodesic(
        initial_point=mean_estimate, initial_tangent_vec=tpca.components_[1]
    )

    n_steps = 100
    t = np.linspace(-1, 1, n_steps)
    geodesic_points_0 = geodesic_0(t)
    geodesic_points_1 = geodesic_1(t)

    logging.info(
        "Coordinates of the Log of the first 5 data points at the mean, "
        "projected on the principal components:"
    )
    logging.info("\n{}".format(tangent_projected_data[:5]))

    ax_var = fig.add_subplot(121)
    xticks = np.arange(1, 2 + 1, 1)
    ax_var.xaxis.set_ticks(xticks)
    ax_var.set_title("Explained variance")
    ax_var.set_xlabel("Number of Principal Components")
    ax_var.set_ylim((0, 1))
    ax_var.plot(xticks, tpca.explained_variance_ratio_)

    ax = fig.add_subplot(122, projection="3d")

    visualization.plot(mean_estimate, ax, space="S2", color="darkgreen", s=10)
    visualization.plot(geodesic_points_0, ax, space="S2", linewidth=2)
    visualization.plot(geodesic_points_1, ax, space="S2", linewidth=2)
    visualization.plot(data, ax, space="S2", color="black", alpha=0.7)

    plt.show()


if __name__ == "__main__":
    main()
