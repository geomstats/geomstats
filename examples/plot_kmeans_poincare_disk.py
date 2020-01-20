"""
Plot a square on H2 with Poincare Disk visualization
with two clusters from uniformly sampled random points.
Then apply the K-means algorithm and plot the points labels
as two distinct colors. Computed means are marked as green stars.
"""
import os

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.kmeans import RiemannianKMeans


def main():

    cluster_1 = gs.random.uniform(low=0.5, high=0.6, size=(20, 2))
    cluster_2 = gs.random.uniform(low=0, high=-0.2, size=(20, 2))

    ax = plt.gca()

    merged_clusters = gs.concatenate((cluster_1, cluster_2), axis=0)
    manifold = Hyperbolic(dimension=2, point_type='ball')
    metric = manifold.metric

    visualization.plot(
            merged_clusters,
            ax=ax,
            space='H2_poincare_disk',
            marker='.',
            color='black',
            point_type=manifold.point_type)

    kmeans = RiemannianKMeans(riemannian_metric=metric,
                              n_clusters=2,
                              init='random',
                              mean_method='frechet-poincare-ball'
                              )

    centroids = kmeans.fit(X=merged_clusters, max_iter=100)

    labels = kmeans.predict(X=merged_clusters)

    visualization.plot(
            merged_clusters[labels == 0],
            ax=ax,
            space='H2_poincare_disk',
            marker='.',
            color='red',
            point_type=manifold.point_type)

    visualization.plot(
            merged_clusters[labels == 1],
            ax=ax,
            space='H2_poincare_disk',
            marker='.',
            color='blue',
            point_type=manifold.point_type)

    visualization.plot(
        centroids,
        ax=ax,
        space='H2_poincare_disk',
        marker='*',
        color='green',
        point_type=manifold.point_type)

    plt.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        print('Examples with visualizations are only implemented '
              'with numpy backend.\n'
              'To change backend, write: '
              'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
