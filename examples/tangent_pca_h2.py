"""Perform tangent PCA at the mean on H2."""

import logging

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.pca import TangentPCA


def main():
    """Perform tangent PCA at the mean on H2."""
    fig = plt.figure(figsize=(15, 5))

    hyperbolic_plane = Hyperboloid(dim=2)

    data = hyperbolic_plane.random_point(n_samples=140)

    mean = FrechetMean(metric=hyperbolic_plane.metric)
    mean.fit(data)

    mean_estimate = mean.estimate_

    tpca = TangentPCA(metric=hyperbolic_plane.metric, n_components=2)
    tpca = tpca.fit(data, base_point=mean_estimate)
    tangent_projected_data = tpca.transform(data)

    geodesic_0 = hyperbolic_plane.metric.geodesic(
        initial_point=mean_estimate,
        initial_tangent_vec=tpca.components_[0])
    geodesic_1 = hyperbolic_plane.metric.geodesic(
        initial_point=mean_estimate,
        initial_tangent_vec=tpca.components_[1])

    n_steps = 100
    t = np.linspace(-1, 1, n_steps)
    geodesic_points_0 = geodesic_0(t)
    geodesic_points_1 = geodesic_1(t)

    logging.info(
        'Coordinates of the Log of the first 5 data points at the mean, '
        'projected on the principal components:')
    logging.info('\n{}'.format(tangent_projected_data[:5]))

    ax_var = fig.add_subplot(121)
    xticks = np.arange(1, 2 + 1, 1)
    ax_var.xaxis.set_ticks(xticks)
    ax_var.set_title('Explained variance')
    ax_var.set_xlabel('Number of Principal Components')
    ax_var.set_ylim((0, 1))
    ax_var.plot(xticks, tpca.explained_variance_ratio_)

    ax = fig.add_subplot(122)

    visualization.plot(
        mean_estimate, ax, space='H2_poincare_disk', color='darkgreen', s=10)
    visualization.plot(
        geodesic_points_0, ax, space='H2_poincare_disk', linewidth=2)
    visualization.plot(
        geodesic_points_1, ax, space='H2_poincare_disk', linewidth=2)
    visualization.plot(
        data, ax, space='H2_poincare_disk', color='black', alpha=0.7)

    plt.show()


if __name__ == '__main__':
    main()
