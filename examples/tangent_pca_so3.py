"""
Compute the mean of a data set of 3D rotations.
Performs tangent PCA at the mean.
"""

import matplotlib.pyplot as plt
import numpy as np

import geomstats.visualization as visualization

from geomstats.special_orthogonal_group import SpecialOrthogonalGroup

SO3_GROUP = SpecialOrthogonalGroup(n=3)
METRIC = SO3_GROUP.bi_invariant_metric

N_SAMPLES = 20


def main():
    fig = plt.figure(figsize=(15, 5))

    data = SO3_GROUP.random_uniform(n_samples=N_SAMPLES)
    mean = METRIC.mean(data)

    eigenvalues, tangent_eigenvecs = METRIC.tangent_pca(data, base_point=mean)
    n_eigenvalues = min(N_SAMPLES, SO3_GROUP.dimension)

    explained_variances = np.cumsum(eigenvalues)
    explained_variances = explained_variances / explained_variances[-1]

    ax_eig = fig.add_subplot(131)
    xticks = np.arange(0, n_eigenvalues, 1)
    ax_eig.xaxis.set_ticks(xticks)
    ax_eig.set_title('Eigenvalues in decreasing order')
    ax_eig.plot(xticks, eigenvalues)

    ax_var = fig.add_subplot(132)
    xticks = np.arange(0, n_eigenvalues, 1)
    ax_var.xaxis.set_ticks(xticks)
    ax_var.set_title('Explained variance')
    ax_var.set_xlabel('Number of Principal Components')
    ax_var.plot(xticks, explained_variances)

    ax = fig.add_subplot(133, projection="3d", aspect="equal")
    plt.setp(ax,
             xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
             xlabel="X", ylabel="Y", zlabel="Z")

    visualization.plot(data, ax, space='SO3_GROUP', color='black')
    visualization.plot(mean, ax, space='SO3_GROUP', linewidth=3)
    plt.show()


if __name__ == "__main__":
    main()
