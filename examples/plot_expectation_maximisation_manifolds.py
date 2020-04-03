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
from geomstats.learning.em_expectation_maximization import distance

PI_2_3 = pow((2 * gs.pi), 2 / 3)
CST_FOR_ERF = 8.0 / (3.0 * gs.pi) * (gs.pi - 3.0) / (4.0 - gs.pi)

def plot_embedding_distribution(data,
                                mixture_coefficients,
                                means,
                                variances, labels=None,
                                plot_precision=100,
                                colors=None,
                                save_path=""):

    x_axis_samples = gs.linspace(-1, 1, plot_precision)
    y_axis_samples = gs.linspace(-1, 1, plot_precision)
    x_axis_samples, y_axis_samples = gs.meshgrid(x_axis_samples,
                                                 y_axis_samples)

    z_axis_samples = gs.zeros((plot_precision, plot_precision))

    for z_index in range(len(z_axis_samples)):

        x_y_plane_mesh_gs = gs.concatenate((gs.expand_dims(x_axis_samples[z_index],-1),
                                           gs.expand_dims(y_axis_samples[z_index],-1)),
                                           axis=-1)

        mesh_probabilities = weighted_gmm_pdf(mixture_coefficients,
                                              torch.from_numpy(x_y_plane_mesh_gs),
                                              means,
                                              variances,
                                              distance)

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
        ax.scatter(data[data_index][0].item(),
                       data[data_index][1].item(),
                       z_circle,
                       c='b',
                       marker='.')


    for means_index in range(len(means)):
        ax.scatter(means[means_index][0].item(),
                   means[means_index][1].item(),
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
    cluster_3 = gs.random.uniform(low=0, high=-0.2, size=(n_samples, dim))
    cluster_3[:,0] = -cluster_3[:,0]

    data = gs.concatenate((cluster_1, cluster_2, cluster_3), axis=0)

    data = torch.from_numpy(data)

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


    plot = plot_embedding_distribution(data,
                                       mixture_coefficients,
                                       means,
                                       variances,
                                       labels=None,
                                       plot_precision=100,
                                       colors=None,
                                       save_path=os.path.join("result.png")
                                       )

    return plot




def erf_approx(x):
    return torch.sign(x)*torch.sqrt(1 - torch.exp(-x * x * (4 / np.pi + CST_FOR_ERF * x * x) / (1 + CST_FOR_ERF * x * x)))

def weighted_gmm_pdf(w, z, mu, sigma, distance):
    # print(z.size())
    # print(z.size(0), len(mu), z.size(1))
    z_u = z.unsqueeze(1).expand(z.size(0), len(mu), z.size(1))
    # print(z_u.size())
    # print(mu.size())
    mu_u = mu.unsqueeze(0).expand_as(z_u)

    distance_to_mean = distance(z_u, mu_u)
    sigma_u = sigma.unsqueeze(0).expand_as(distance_to_mean)
    distribution_normal = torch.exp(-((distance_to_mean)**2)/(2 * sigma_u**2))
    zeta_sigma = PI_2_3 * sigma * torch.exp((sigma ** 2 / 2) * erf_approx(sigma / math.sqrt(2)))

    return w.unsqueeze(0).expand_as(distribution_normal) * distribution_normal/zeta_sigma.unsqueeze(0).expand_as(distribution_normal)

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
