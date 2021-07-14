"""Run K-means on manifolds for K=2 and Plot the results.

Two random clusters are generated in separate regions of the
manifold. Then K-means is applied using the metric of the manifold.
The points are represented with two distinct colors. For the moment
the example works on the Poincaré Ball and the Hypersphere.
Computed means are marked as green stars.
"""

import logging
import os

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.poincare_ball import PoincareBall
from geomstats.learning.kmeans import RiemannianKMeans


def kmean_poincare_ball():
    """Run K-means on the Poincare ball."""
    n_samples = 20
    dim = 2
    n_clusters = 2
    manifold = PoincareBall(dim=dim)
    metric = manifold.metric

    cluster_1 = gs.random.uniform(low=0.5, high=0.6, size=(n_samples, dim))
    cluster_2 = gs.random.uniform(low=0, high=-0.2, size=(n_samples, dim))
    data = gs.concatenate((cluster_1, cluster_2), axis=0)

    kmeans = RiemannianKMeans(metric=metric,
                              n_clusters=n_clusters,
                              init='random')

    centroids = kmeans.fit(X=data)
    labels = kmeans.predict(X=data)

    plt.figure(1)
    colors = ['red', 'blue']

    ax = visualization.plot(
        data,
        space='H2_poincare_disk',
        marker='.',
        color='black',
        point_type=manifold.point_type)

    for i in range(n_clusters):
        ax = visualization.plot(
            data[labels == i],
            ax=ax,
            space='H2_poincare_disk',
            marker='.',
            color=colors[i],
            point_type=manifold.point_type)

    ax = visualization.plot(
        centroids,
        ax=ax,
        space='H2_poincare_disk',
        marker='*',
        color='green',
        s=100,
        point_type=manifold.point_type)

    ax.set_title('Kmeans on Poincaré Ball Manifold')

    return plt


def kmean_hypersphere():
    """Run K-means on the sphere."""
    n_samples = 50
    dim = 2
    n_clusters = 2
    manifold = Hypersphere(dim)
    metric = manifold.metric

    # Generate data on north pole
    cluster_1 = manifold.random_von_mises_fisher(kappa=50, n_samples=n_samples)

    # Generate data on south pole
    cluster_2 = manifold.random_von_mises_fisher(kappa=50, n_samples=n_samples)
    cluster_2 = - cluster_2

    data = gs.concatenate((cluster_1, cluster_2), axis=0)

    kmeans = RiemannianKMeans(metric, n_clusters, tol=1e-3)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.centroids

    plt.figure(2)
    colors = ['red', 'blue']

    ax = visualization.plot(
        data,
        space='S2',
        marker='.',
        color='black')

    for i in range(n_clusters):
        if len(data[labels == i]) > 0:
            ax = visualization.plot(
                points=data[labels == i],
                ax=ax,
                space='S2',
                marker='.',
                color=colors[i])

    ax = visualization.plot(
        centroids,
        ax=ax,
        space='S2',
        marker='*',
        s=200,
        color='green')

    ax.set_title('Kmeans on the sphere')

    return plt


def main():
    """Run K-means on the Poincare ball and the sphere."""
    kmean_poincare_ball()

    plots = kmean_hypersphere()

    plots.show()


if __name__ == '__main__':
    if os.environ['GEOMSTATS_BACKEND'] != 'numpy':
        logging.info('Examples with visualizations are only implemented '
                     'with numpy backend.\n'
                     'To change backend, write: '
                     'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
