"""
Applies Expectation Maximization on manifolds and plots the results.

Two random clusters are generated in seperate regions of the
manifold. Then apply EM using the metric of the manifold
algorithm and plot the Gaussian Mixture Model. For the moment
the example works on the Poincaré Ball.
"""
import os

import matplotlib.pyplot as plt
import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.learning.em_expectation_maximization import RiemannianEM,RawDataloader
import numpy as np
import math
import torch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle

PI_2_3 = pow((2 * gs.pi), 2 / 3)
CST_FOR_ERF = 8.0 / (3.0 * gs.pi) * (gs.pi - 3.0) / (4.0 - gs.pi)

def plot_gaussian_mixture_distribution(data,
                                mixture_coefficients,
                                means,
                                variances,
                                plot_precision=10,
                                save_path="",
                                metric=None):

    x_axis_samples = gs.linspace(-1, 1, plot_precision)
    y_axis_samples = gs.linspace(-1, 1, plot_precision)
    x_axis_samples, y_axis_samples = gs.meshgrid(x_axis_samples,
                                                 y_axis_samples)

    z_axis_samples = gs.zeros((plot_precision, plot_precision))

    for z_index in range(len(z_axis_samples)):

        x_y_plane_mesh = gs.concatenate((gs.expand_dims(x_axis_samples[z_index],-1),
                                           gs.expand_dims(y_axis_samples[z_index],-1)),
                                           axis=-1)

        mesh_probabilities = weighted_gmm_pdf(mixture_coefficients,
                                              x_y_plane_mesh,
                                              means,
                                              variances,
                                              metric)

        mesh_probabilities[mesh_probabilities != mesh_probabilities ]= 0

        z_axis_samples[z_index] = mesh_probabilities.sum(-1)

    fig = plt.figure("Learned Gaussian Mixture Model via Expectation Maximisation on Poincaré Disc")

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
                             z = z_circle,
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

    #Generate random data in 3 different parts of the manifold

    dim = 2
    n_samples = 5

    cluster_1 = gs.random.uniform(low=0.2, high=0.6, size=(n_samples, dim))
    cluster_2 = gs.random.uniform(low=-0.2, high=-0.6, size=(n_samples, dim))
    cluster_3 = gs.random.uniform(low=0, high=-0.3, size=(n_samples, dim))
    cluster_3[:,0] = -cluster_3[:,0]

    data = gs.concatenate((cluster_1, cluster_2, cluster_3), axis=0)



    n_clusters = 3

    manifold = Hyperbolic(dimension=2, coords_type='ball')

    metric = manifold.metric

    EM = RiemannianEM(riemannian_metric=metric,
                       n_gaussian= n_clusters,
                       init='random',
                       mean_method='frechet-poincare-ball',
                       verbose=1,
                       )

    means, variances, mixture_coefficients = EM.fit(
        data=data,
        max_iter=100)


    plot = plot_gaussian_mixture_distribution(data,
                                       mixture_coefficients.data.numpy(),
                                       means.data.numpy(),
                                       variances.data.numpy(),
                                       plot_precision=100,
                                       save_path=os.path.join("result.png"),
                                       metric = metric.dist
                                       )

    return plot

def erf_approx(x):
    return gs.sign(x)*gs.sqrt(1 - gs.exp(-x * x * (4 / gs.pi + CST_FOR_ERF * x * x) / (1 + CST_FOR_ERF * x * x)))

def weighted_gmm_pdf(mixture_coefficients,
                     mesh_data,
                     means,
                     variances,
                     metric):

    mesh_data_units = gs.expand_dims(mesh_data, 1)

    mesh_data_units = gs.repeat(mesh_data_units, len(means), axis = 1)

    means_units = gs.expand_dims(means,0)

    means_units = gs.repeat(means_units,mesh_data_units.shape[0],axis = 0)

    distance_to_mean = metric(mesh_data_units, means_units)
    variances_units = gs.expand_dims(variances,0)
    variances_units = gs.repeat(variances_units, distance_to_mean.shape[0], axis = 0)

    distribution_normal = gs.exp(-((distance_to_mean)**2)/(2 * variances_units**2))

    zeta_sigma =PI_2_3 * variances * gs.exp((variances ** 2 / 2) * erf_approx(variances / gs.sqrt(2)))

    result_num_gs = gs.expand_dims(mixture_coefficients,0)
    result_num_gs = gs.repeat(result_num_gs,len(distribution_normal), axis = 0) * distribution_normal
    result_denum_gs = gs.expand_dims(zeta_sigma,0)
    result_denum_gs = gs.repeat(result_denum_gs,len(distribution_normal), axis = 0)

    result = result_num_gs/result_denum_gs

    return result

def main():

    # Expectation Maximisation Poincare Ball
    plots = expectation_maximisation_poincare_ball()

    plots.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] != 'numpy':
        print('Expectation Maximization example\n'
              'works with\n'
              'with numpy backend.\n'
              'To change backend, write: '
              'export GEOMSTATS_BACKEND = \'numpy\'.')
    else:
        main()
