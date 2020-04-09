"""
Applies Expectation Maximization on manifolds and plots the results.

Random data is generated in seperate regions of the
manifold. Then Expectation Maximization deduces a Gaussian Mixture Model
that best fits the random data. For the moment
the example works on the Poincaré Ball hyperbolic space.
"""

import os

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.gaussian_distribution import GaussianDistribution
from geomstats.learning.em_expectation_maximization import RiemannianEM

DEFAULT_PLOT_PRECISION = 100


def plot_gaussian_mixture_distribution(data,
                                       mixture_coefficients,
                                       means,
                                       variances,
                                       plot_precision=DEFAULT_PLOT_PRECISION,
                                       save_path="",
                                       metric=None):
    """Plot Gaussian Mixture Model."""
    x_axis_samples = gs.linspace(-1, 1, plot_precision)
    y_axis_samples = gs.linspace(-1, 1, plot_precision)
    x_axis_samples, y_axis_samples = gs.meshgrid(x_axis_samples,
                                                 y_axis_samples)

    z_axis_samples = gs.zeros((plot_precision, plot_precision))

    for z_index in range(len(z_axis_samples)):

        x_y_plane_mesh = gs.concatenate((
            gs.expand_dims(x_axis_samples[z_index], -1),
            gs.expand_dims(y_axis_samples[z_index], -1)),
            axis=-1)

        mesh_probabilities = GaussianDistribution.weighted_gmm_pdf(mixture_coefficients,
                                              x_y_plane_mesh,
                                              means,
                                              variances,
                                              metric)

        mesh_probabilities[mesh_probabilities != mesh_probabilities] = 0

        z_axis_samples[z_index] = mesh_probabilities.sum(-1)

    fig = plt.figure("Learned Gaussian Mixture Model "
                     "via Expectation Maximisation on Poincaré Disc")

    ax = fig.gca(projection='3d')
    ax.plot_surface(x_axis_samples,
                    y_axis_samples,
                    z_axis_samples,
                    rstride=1,
                    cstride=1,
                    linewidth=1,
                    antialiased=True,
                    cmap=plt.get_cmap("viridis"))
    z_circle = -0.8
    p = Circle((0, 0), 1,
               edgecolor='b',
               lw=1,
               facecolor='none')

    ax.add_patch(p)

    art3d.pathpatch_2d_to_3d(p,
                             z=z_circle,
                             zdir="z")

    for data_index in range(len(data)):
        ax.scatter(data[data_index][0],
                   data[data_index][1],
                   z_circle,
                   c='b',
                   marker='.')

    for means_index in range(len(means)):
        ax.scatter(means[means_index][0],
                   means[means_index][1],
                   z_circle,
                   c='r',
                   marker='D')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.8, 0.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')

    plt.savefig(save_path, format="pdf")

    return plt


def expectation_maximisation_poincare_ball():
    """Apply EM algorithm on three random data clusters."""
    dim = 2
    n_samples = 5

    cluster_1 = gs.random.uniform(low=0.2, high=0.6, size=(n_samples, dim))
    cluster_2 = gs.random.uniform(low=-0.2, high=-0.6, size=(n_samples, dim))
    cluster_3 = gs.random.uniform(low=0, high=-0.3, size=(n_samples, dim))
    cluster_3[:, 0] = -cluster_3[:, 0]

    data = gs.concatenate((cluster_1, cluster_2, cluster_3), axis=0)

    # Set up hyperbolic space

    n_clusters = 3

    manifold = Hyperbolic(dimension=2, coords_type='ball')

    metric = manifold.metric

    EM = RiemannianEM(n_gaussian=n_clusters,
                      riemannian_metric=metric,
                      initialisation_method='random',
                      mean_method='frechet-poincare-ball',
                      verbose=1)

    means, variances, mixture_coefficients = EM.fit(
        data=data,
        max_iter=100)

    # Plot result
    plot = plot_gaussian_mixture_distribution(data,
                                              mixture_coefficients,
                                              means,
                                              variances,
                                              plot_precision=100,
                                              save_path='result.png',
                                              metric=metric.dist)

    return plot


def main():
    """Define main function."""
    plots = expectation_maximisation_poincare_ball()

    plots.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] != 'numpy':
        print('Expectation Maximization example\n'
              'works with\n'
              'numpy backend.\n'
              'To change backend, write: '
              'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
