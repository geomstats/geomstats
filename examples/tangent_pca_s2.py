"""
Compute the mean of a data set on the sphere.
Performs tangent PCA at the mean.
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings

import geomstats.visualization as visualization

from geomstats.learning.pca import TangentPCA
from geomstats.hypersphere import Hypersphere


def main():
    fig = plt.figure(figsize=(15, 5))

    sphere = Hypersphere(dimension=2)

    data = sphere.random_von_mises_fisher(kappa=15, n_samples=140)
    mean = sphere.metric.mean(data)

    tpca = TangentPCA(metric=sphere.metric, n_components=2)
    tpca = tpca.fit(data, base_point=mean)
    tangent_projected_data = tpca.transform(data)

    geodesic_0 = sphere.metric.geodesic(
        initial_point=mean,
        initial_tangent_vec=tpca.components_[0])
    geodesic_1 = sphere.metric.geodesic(
        initial_point=mean,
        initial_tangent_vec=tpca.components_[1])

    n_steps = 100
    t = np.linspace(-1, 1, n_steps)
    geodesic_points_0 = geodesic_0(t)
    geodesic_points_1 = geodesic_1(t)

    print(
        'Coordinates of the Log of the first 5 data points at the mean, '
        'projected on the principal components:')
    print(tangent_projected_data[:5])

    ax_var = fig.add_subplot(121)
    xticks = np.arange(1, 2+1, 1)
    ax_var.xaxis.set_ticks(xticks)
    ax_var.set_title('Explained variance')
    ax_var.set_xlabel('Number of Principal Components')
    ax_var.set_ylim((0, 1))
    ax_var.plot(xticks, tpca.explained_variance_ratio_)

    ax = fig.add_subplot(122, projection="3d")

    visualization.plot(
        mean, ax, space='S2', color='darkgreen', s=10)
    visualization.plot(
        geodesic_points_0, ax, space='S2', linewidth=2)
    visualization.plot(
        geodesic_points_1, ax, space='S2', linewidth=2)
    visualization.plot(
        data, ax, space='S2', color='green', alpha=0.7)

    plt.show()


if __name__ == "__main__":
    warnings.simplefilter('ignore')
    main()
