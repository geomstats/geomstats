"""Perform tangent PCA at the mean on SO(3)."""

import logging

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA

SO3_GROUP = SpecialOrthogonal(n=3, point_type="vector")
METRIC = SO3_GROUP.bi_invariant_metric

N_SAMPLES = 10
N_COMPONENTS = 2


def main():
    """Perform tangent PCA at the mean on SO(3)."""
    fig = plt.figure(figsize=(15, 5))

    data = SO3_GROUP.random_uniform(n_samples=N_SAMPLES)

    mean = FrechetMean(metric=METRIC)
    mean.fit(data)

    mean_estimate = mean.estimate_

    tpca = TangentPCA(metric=METRIC, n_components=N_COMPONENTS)
    tpca = tpca.fit(data, base_point=mean_estimate)
    tangent_projected_data = tpca.transform(data)
    logging.info(
        "Coordinates of the Log of the first 5 data points at the mean, "
        "projected on the principal components:"
    )
    logging.info("\n{}".format(tangent_projected_data[:5]))

    ax_var = fig.add_subplot(121)
    xticks = np.arange(1, N_COMPONENTS + 1, 1)
    ax_var.xaxis.set_ticks(xticks)
    ax_var.set_title("Explained variance")
    ax_var.set_xlabel("Number of Principal Components")
    ax_var.set_ylim((0, 1))
    ax_var.plot(xticks, tpca.explained_variance_ratio_)

    ax = fig.add_subplot(122, projection="3d")
    plt.setp(ax, xlabel="X", ylabel="Y", zlabel="Z")

    ax.set_title("Data in SO3 (black) and Frechet mean (color)")
    visualization.plot(data, ax, space="SO3_GROUP", color="black")
    visualization.plot(mean_estimate, ax, space="SO3_GROUP", linewidth=3)
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_zlim((-2, 2))
    plt.show()


if __name__ == "__main__":
    main()
