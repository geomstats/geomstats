"""
Applies K-means on manifolds and plots the results.

Two random clusters are generated in seperate regions of the
manifold. Then apply K-means is applied using the metric of the manifold
algorithm and plot the points labels as two distinct colors. For the moment
the example works on the Poincaré Ball and the Hypersphere.
Computed means are marked as green stars.
"""

import logging
import os

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kmeans import RiemannianKMeans


def kmean_poincare_ball():

    n_samples = 20
    dim = 2
    n_clusters = 2
    manifold = Hyperbolic(dimension=dim, point_type='ball')
    metric = manifold.metric

    cluster_1 = gs.random.uniform(low=0.5, high=0.6, size=(n_samples, dim))
    cluster_2 = gs.random.uniform(low=0, high=-0.2, size=(n_samples, dim))
    data = gs.concatenate((cluster_1, cluster_2), axis=0)

    kmeans = RiemannianKMeans(riemannian_metric=metric,
                              n_clusters=n_clusters,
                              init='random',
                              mean_method='frechet-poincare-ball'
                              )

    centroids = kmeans.fit(X=data, max_iter=100)
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

    n_samples = 50
    dim = 2
    n_clusters = 2
    manifold = Hypersphere(dim)
    metric = manifold.metric

    # Generate data on north pole
    cluster_1 = manifold.random_von_mises_fisher(kappa=50, n_samples=n_samples)

    # Generate data on south pole
    cluster_2 = manifold.random_von_mises_fisher(kappa=50, n_samples=n_samples)
    for point in cluster_2:
        point[2] = -point[2]

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
        ax = visualization.plot(
            data[labels == i],
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

    ax.set_title('Kmeans on Hypersphere Manifold')

    return plt


def main():

    kmean_poincare_ball()

    plots = kmean_hypersphere()

    plots.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] != 'numpy':
        logging.info('Examples with visualizations are only implemented '
                     'with numpy backend.\n'
                     'To change backend, write: '
                     'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
