"""
Plot a square on H2 with Poincare Disk visualization
with two clusters from uniformly sampled random points.
Then apply the K-means algorithm and plot the points labels
as two distinct colors.
"""
import os

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.geometry.hyperbolic_space import HyperbolicSpace
from geomstats.learning.k_means import RiemannianKMeans

H2 = HyperbolicSpace(dimension=2, point_type='poincare')
METRIC = H2.metric

SQUARE_SIZE = 50


def main():

    Cluster_1 = gs.random.uniform(low=0.5, high=0.6, size=(20, 2))
    Cluster_2 = gs.random.uniform(low=0, high=-0.2, size=(20, 2))

    ax = plt.gca()

    Merged_Clusters = gs.concatenate((Cluster_1, Cluster_2), axis=0)

    visualization.plot(
            Merged_Clusters,
            ax=ax,
            space='H2_poincare_disk',
            marker='.',
            color='black',
            point_type=H2.point_type)

    K_means = RiemannianKMeans(riemannian_metric=H2.metric,
                               n_clusters=2,
                               init='random',
                               )

    Centroids = K_means.fit(X=Merged_Clusters, max_iter=5)

    Data_Labels = gs.array([])

    for data in Merged_Clusters:

        Data_Labels = gs.append(Data_Labels, K_means.predict(data))

    print('Centroids', Centroids)

    visualization.plot(
            Centroids,
            ax=ax,
            space='H2_poincare_disk',
            marker='.',
            color='red',
            point_type=H2.point_type)

    print('Data_labels', Data_Labels)

    # visualization.plot(
    #         Merged_Clusters[],
    #         ax=ax,
    #         space='H2_poincare_disk',
    #         marker='.',
    #         color='red',
    #         point_type=H2.point_type)
    plt.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] == 'tensorflow':
        print('Examples with visualizations are only implemented '
              'with numpy backend.\n'
              'To change backend, write: '
              'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
