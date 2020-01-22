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
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kmeans import RiemannianKMeans


def kmean_poincare_ball(data, n_clusters = 2):

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
        point_type=manifold.point_type)

    return plt

def kmean_hypersphere(data, n_clusters=2):

    plt.figure(2)
    ax = plt.gca()

    manifold = Hypersphere(2)
    metric = manifold.metric


    kmeans = RiemannianKMeans(metric, 1, tol=1e-3)
    kmeans.fit(data)
    center = kmeans.centroids
    mean = metric.mean(data)
    result = metric.dist(center, mean)

    print('result')
    return plt


def main():

    cluster_1 = gs.random.uniform(low=0.5, high=0.6, size=(20, 2))
    cluster_2 = gs.random.uniform(low=0, high=-0.2, size=(20, 2))

    merged_clusters = gs.concatenate((cluster_1, cluster_2), axis=0)
    n_clusters = 2


    #Kmean Poincare Ball
    #plot_poincare = kmean_poincare_ball(merged_clusters, n_clusters)

    #Kmean Euclidean
    plot_hypersphere = kmean_hypersphere(merged_clusters, n_clusters)

    plot_hypersphere.show()






if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        print('Examples with visualizations are only implemented '
              'with numpy backend.\n'
              'To change backend, write: '
              'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
