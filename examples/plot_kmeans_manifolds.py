"""
Applies K-means on manifolds and plots the results.

Two random clusters are generated in seperate regions of the
manifold. Then apply K-means is applied using the metric of the manifold
algorithm and plot the points labels as two distinct colors. For the moment
the example works on the Poincaré Ball and the Hypersphere.
Computed means are marked as green stars.
"""
import os

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kmeans import RiemannianKMeans


def kmean_poincare_ball():

    cluster_1 = gs.random.uniform(low=0.5, high=0.6, size=(20, 2))
    cluster_2 = gs.random.uniform(low=0, high=-0.2, size=(20, 2))

    data = gs.concatenate((cluster_1, cluster_2), axis=0)
    n_clusters = 2

    plt.figure(1)

    ax = plt.gca()

    manifold = Hyperbolic(dimension=2, point_type='ball')
    metric = manifold.metric

    visualization.plot(
        data,
        ax=ax,
        space='H2_poincare_disk',
        marker='.',
        color='black',
        point_type=manifold.point_type)

    kmeans = RiemannianKMeans(riemannian_metric=metric,
                              n_clusters=n_clusters,
                              init='random',
                              mean_method='frechet-poincare-ball'
                              )

    centroids = kmeans.fit(X=data, max_iter=100)

    labels = kmeans.predict(X=data)

    colors = ['red', 'blue']

    for i in range(n_clusters):

        visualization.plot(
            data[labels == i],
            ax=ax,
            space='H2_poincare_disk',
            marker='.',
            color=colors[i],
            point_type=manifold.point_type)

    visualization.plot(
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

    plt.figure(2)

    manifold = Hypersphere(2)
    metric = manifold.metric

    # Generate data on north pole
    cluster_1 = manifold.random_von_mises_fisher(kappa=50, n_samples=50)

    # Generate data on south pole
    cluster_2 = manifold.random_von_mises_fisher(kappa=50, n_samples=50)
    for point in cluster_2:
        point[2] = -point[2]

    data = gs.concatenate((cluster_1, cluster_2), axis=0)

    kmeans = RiemannianKMeans(metric, 2, tol=1e-3)
    kmeans.fit(data)
    labels = kmeans.predict(data)
    centroids = kmeans.centroids

    colors = ['red', 'blue']

    for i in range(2):

        if i == 0:

            ax = visualization.plot(
                data[labels == i],
                space='S2',
                marker='.',
                color=colors[i]
            )
        else:
            ax = visualization.plot(
                data[labels == i],
                ax=ax,
                space='S2',
                marker='.',
                color=colors[i]
            )

    visualization.plot(
        centroids,
        ax=ax,
        space='S2',
        marker='*',
        s=200,
        color='green',
    )

    ax.set_title('Kmeans on Hypersphere Manifold')

    return plt


def main():

    # Kmean Poincare Ball
    kmean_poincare_ball()

    # Kmean Hypersphere
    plots = kmean_hypersphere()

    plots.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        print('Examples with visualizations are only implemented '
              'with numpy backend.\n'
              'To change backend, write: '
              'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
