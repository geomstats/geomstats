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
import geomstats.visualization as visualization
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.kmeans import RiemannianKMeans
from geomstats.learning.em_expectation_maximization import RiemannianEM,RawDataloader
import numpy as np
import math
import torch
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle
from geomstats.learning.em_expectation_maximization import distance


def plot_embedding_distribution(W, pi, mu, sigma,  labels=None, N=100, colors=None, save_path=""):

    # TODO : labels colors
    if(labels is None):
        # color depending from gaussian prob
        pass
    else:
        # color depending from labels given
        pass

    # plotting prior
    X = np.linspace(-1, 1 ,N)
    Y = np.linspace(-1, 1 ,N)
    X, Y = np.meshgrid(X, Y)
    # plotting circle
    X0, Y0, radius = 0, 0, 1
    r = np.sqrt((X - X0)**2 + (Y * Y0)**2)
    disc = r < 1

    Z = np.zeros((N, N))
    # compute the mixture
    for z_index in range(len(Z)):
        #    print(torch.Tensor(X[z_index]))
        x =  torch.cat((torch.FloatTensor(X[z_index]).unsqueeze(-1), torch.FloatTensor(Y[z_index]).unsqueeze(-1)), -1)
        zz = weighted_gmm_pdf(pi, x, mu, sigma, distance)
        zz[zz != zz ]= 0
        #    print(zz.size())
        #    print(zz)
        #    print(weighted_gmm_pdf(pi, mu, mu, sigma, pf.distance))
        Z[z_index] = zz.sum(-1).numpy()
    # print(Z.max())
    fig = plt.figure("Embedding-Distribution")
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True, cmap=plt.get_cmap("viridis"))
    z_circle = -0.8
    p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z = z_circle, zdir="z")

    for q in range(len(W)):
        if(colors is not None):
            ax.scatter(W[q][0].item(), W[q][1].item(), z_circle, c=[colors[q]], marker='.')
        else:
            ax.scatter(W[q][0].item(), W[q][1].item(), z_circle, c='b', marker='.')
        #print('Print labels', labels[q])

    for j in range(len(mu)):
        ax.scatter(mu[j][0].item(), mu[j][1].item(), z_circle, c='r', marker='D')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.8, 0.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')

    plt.savefig(save_path, format="png")



def expectation_maximisation_poincare_ball():

    # cluster_1 = gs.random.rand(low=0.2, high=0.6, size=(20, 2))
    # cluster_2 = gs.random.rand(low=0, high=-0.5, size=(20, 2))
    # cluster_3 = gs.random.rand(low=0.2, high=0.6, size=(20, 2))

    r1 = 0.2
    r2 = 0.5
    r3 = 0
    r4 = -0.5
    r5 = 0.6
    cluster_1 = (0.3)* gs.random.rand(20, 2) +0.3
    cluster_2 = -(0.5)* gs.random.rand(20, 2)-0.2
    cluster_3 = (0.4)* gs.random.rand(20, 2)
    cluster_3[:,0] = -cluster_3[:,0]

    data = gs.concatenate((cluster_1, cluster_2, cluster_3), axis=0)

    n_clusters = 3

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

    EM = RiemannianEM (riemannian_metric=metric,
                       n_gaussian= n_clusters,
                       init='random',
                       mean_method='frechet-poincare-ball',
                       verbose=1,
                       )

    # dataset_o1 = corpora.NeigbhorFlatCorpus(X, Y)
    #
    # batch_size = 10000
    #
    # training_dataloader_o1 = data.RawDataloader(dataset_o1,
    #                             batch_size=batch_size*10,
    #                             shuffle=True
    #                     )

    mu, sigma, pi = EM.fit(
        data=data,
        max_iter=100)


    print('mu', mu)
    print('sigma', sigma)
    print('pi', pi)


    # plot_embedding_distribution(data,
    #                                              [pi.cpu()], [mu.cpu()], [sigma.cpu()],
    #                                              labels=None, N=100, colors=None,
    #                                              save_path=os.path.join("result.png")
    #                                              )

    plot_embedding_distribution(data, pi, mu, sigma,
                                                 labels=None, N=100, colors=None,
                                                 save_path=os.path.join("result.png")
                                                 )

    # labels = kmeans.predict(X=data)
    #
    # colors = ['red', 'blue']
    #
    # for i in range(n_clusters):
    #
    #     visualization.plot(
    #         data[labels == i],
    #         ax=ax,
    #         space='H2_poincare_disk',
    #         marker='.',
    #         color=colors[i],
    #         point_type=manifold.point_type)
    #
    # visualization.plot(
    #     centroids,
    #     ax=ax,
    #     space='H2_poincare_disk',
    #     marker='*',
    #     color='green',
    #     s=100,
    #     point_type=manifold.point_type)

    ax.set_title('Expectation Maximisation on Poincaré Ball Manifold')

    return plt


def plot_embedding_distribution_multi(W, pi, mu, sigma, labels=None, N=100, colors=None, save_path="figures/default.pdf"):

    fig = plt.figure("Embedding-Distribution")
    border_size = (math.sqrt(len(W)+0.0))
    if(border_size != round(border_size)):
        border_size += 1
    for i in range(1):
        ax = fig.add_subplot(border_size, border_size, i+1, projection='3d')
        #subplot_embedding_distribution(ax, W[i], pi[i], mu[i], sigma[i], labels=labels, N=N, colors=colors)
        subplot_embedding_distribution(ax, pi[i], mu[i], sigma[i], labels=labels, N=N, colors=colors)
    plt.savefig(save_path, format="png")

    return fig

def subplot_embedding_distribution(ax, pi, mu, sigma,  labels=None, N=100, colors=None ):
    # plotting prior
    X = np.linspace(-1, 1 ,N)
    Y = np.linspace(-1, 1 ,N)
    X, Y = np.meshgrid(X, Y)
    # plotting circle
    X0, Y0, radius = 0, 0, 1
    r = np.sqrt((X - X0)**2 + (Y * Y0)**2)
    disc = r < 1

    Z = np.zeros((N, N))
    # compute the mixture
    for z_index in range(len(Z)):
        x =  torch.cat((torch.FloatTensor(X[z_index]).unsqueeze(-1), torch.FloatTensor(Y[z_index]).unsqueeze(-1)), -1)
        zz = weighted_gmm_pdf(pi, x, mu, sigma, distance)
        zz[zz != zz ]= 0
        Z[z_index] = zz.sum(-1).numpy()

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True, cmap=plt.get_cmap("viridis"))
    z_circle = -0.8
    p = Circle((0, 0), 1, edgecolor='b', lw=1, facecolor='none')
    ax.add_patch(p)
    art3d.pathpatch_2d_to_3d(p, z = z_circle, zdir="z")

    # for q in range(len(W)):
    #     if(colors is not None):
    #         ax.scatter(W[q][0].item(), W[q][1].item(), z_circle, c=[colors[q]], marker='.')
    #     else:
    #         ax.scatter(W[q][0].item(), W[q][1].item(), z_circle, c='b', marker='.')
        #print('Print labels', labels[q])

    for j in range(len(mu)):
        ax.scatter(mu[j][0].item(), mu[j][1].item(), z_circle, c='r', marker='D')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-0.8, 0.4)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('P')

pi_2_3 = pow((2*math.pi),2/3)
a_for_erf = 8.0/(3.0*np.pi)*(np.pi-3.0)/(4.0-np.pi)

def erf_approx(x):
    return torch.sign(x)*torch.sqrt(1-torch.exp(-x*x*(4/np.pi+a_for_erf*x*x)/(1+a_for_erf*x*x)))

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
    zeta_sigma = pi_2_3 * sigma *  torch.exp((sigma**2/2) * erf_approx(sigma/math.sqrt(2)))

    return w.unsqueeze(0).expand_as(distribution_normal) * distribution_normal/zeta_sigma.unsqueeze(0).expand_as(distribution_normal)

def main():

    # Expectation Maximisation Poincare Ball
    plots = expectation_maximisation_poincare_ball()

    plots.show()


if __name__ == "__main__":
    if os.environ['GEOMSTATS_BACKEND'] != 'pytorch':
        print('Expectation Maximization example\n'
              'works with\n'
              'with pytorch backend.\n'
              'To change backend, write: '
              'export GEOMSTATS_BACKEND = \'pytorch\'.')
    else:
        main()
