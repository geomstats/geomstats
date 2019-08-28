"""
Compute the mean of a data set of 3D rotations.
Performs tangent PCA at the mean.
"""

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization

from geomstats.learning.pca import TangentPCA
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

SO3_GROUP = SpecialOrthogonalGroup(n=3)
METRIC = SO3_GROUP.bi_invariant_metric

N_SAMPLES = 10
N_COMPONENTS = 2


def main():
    fig = plt.figure(figsize=(15, 5))

    data = SO3_GROUP.random_uniform(n_samples=N_SAMPLES)
    mean = METRIC.mean(data)

    tpca = TangentPCA(metric=METRIC, n_components=N_COMPONENTS)
    tpca = tpca.fit(data, base_point=mean)
    tangent_projected_data = tpca.transform(data)
    print(
        'Coordinates of the Log of the first 5 data points at the mean, '
        'projected on the principal components:')
    print(tangent_projected_data[:5])

    ax_var = fig.add_subplot(121)
    xticks = np.arange(1, N_COMPONENTS+1, 1)
    ax_var.xaxis.set_ticks(xticks)
    ax_var.set_title('Explained variance')
    ax_var.set_xlabel('Number of Principal Components')
    ax_var.set_ylim((0, 1))
    ax_var.plot(xticks, tpca.explained_variance_ratio_)

    ax = fig.add_subplot(122, projection="3d")
    plt.setp(ax, xlabel="X", ylabel="Y", zlabel="Z")

    ax.set_title('Data in SO3 (black) and Frechet mean (color)')
    visualization.plot(data, ax, space='SO3_GROUP', color='black')
    visualization.plot(mean, ax, space='SO3_GROUP', linewidth=3)
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_zlim((-2, 2))
    plt.show()


if __name__ == "__main__":
    main()
