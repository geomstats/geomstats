"""Visualise the uncertainty of empirical Fréchet mean on the Hyperbolic space.

The variance of the Fréchet mean FM_n of a sample of n IID random variables
of variance Var is decreasing more quickly in a hyperbolic space than in a 
Euclidean space. This example computes the  modulation factor
     alpha = Var( FM_n) / ( n * Var)
for isotropic distributions on hyper-spheres of radius 0 < theta < ???? in
the hyperbolic space H_dim (called here a bubble).
"""

import sys
# path for geomstats
project_dir = '../../..'
sys.path.append(project_dir)

import matplotlib.pyplot as plt
import numpy as np

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere


def empirical_frechet_var_bubble(n_sample, theta, dim,
                                 n_expectation=1000):
    """Variance of the empirical Fréchet mean for a bubble distribution.

    Draw n_samples from a bubble distribution, computes its empirical
    Fréchet mean and the square distance to the asymptotic mean. This
    is repeated n_expectation times to compute an approximation of its
    expectation (i.e. its variance) by sampling.

    The bubble distribution is an isotropic distributions on a Riemannian
    hyper sub-sphere of radius 0 < theta < Pi around the north pole of the
    sphere of dimension dim.

    Parameters
    ----------
    n_sample: number of samples to draw
    theta: radius of the bubble distribution
    dim: dimension of the sphere (embedded in R^{dim+1})
    n_expectation: number of computations for approximating the expectation

    Returns
    -------
    tuple (variance, std-dev on the computed variance)
    """
    assert dim > 1, "Dim > 1 needed to draw a uniform sample on sub-sphere"
    var = []
    sphere = Hypersphere(dimension=dim)
    bubble = Hypersphere(dimension=dim - 1)

    # Define north pole
    north_pole = np.zeros(dim + 1)
    north_pole[dim] = 1.0
    for k in range(n_expectation):
        # Sample n points from the uniform distribution on a sub-sphere
        # of radius theta (i.e cos(theta) in ambient space)
        data = gs.zeros((n_sample, dim + 1), dtype=gs.float64)
        directions = bubble.random_uniform(n_sample)
        for i in range(n_sample):
            for j in range(dim):
                data[i, j] = gs.sin(theta) * directions[i, j]
            data[i, dim] = gs.cos(theta)
        current_mean = sphere.metric.adaptive_gradientdescent_mean(
            data, n_max_iterations=64, init_points=[north_pole])
        var.append(sphere.metric.squared_dist(north_pole, current_mean))
    return np.mean(var), 2 * np.std(var) / np.sqrt(n_expectation)


def modulation_factor(n_sample, theta, dim, n_expectation=1000):
    """Modulation factor on the convergence of the empirical Fréchet mean.

    The modulation factor is the ratio of the variance of the empirical
    Fréchet mean on the manifold to the variance in a Euclidean space,
    for n_samples drawn from an isotropic distributions on a Riemannian
    hyper sub-sphere of radius 0 < theta < Pi around the north pole of the
    sphere of dimension dim.

    Parameters
    ----------
    n_sample: number of samples to draw
    theta: radius of the bubble distribution
    dim: dimension of the sphere (embedded in R^{dim+1})
    n_expectation: number of computations for approximating the expectation

    Returns
    -------
    tuple (modulation factor, std-dev on the modulation factor)
    """
    (var, std_var) = empirical_frechet_var_bubble(
        n_sample, theta, dim, n_expectation=n_expectation)
    return var * n_sample / theta ** 2, std_var * n_sample / theta ** 2


def asymptotic_modulation(dim, theta):
    """Compute the asymptotic modulation factor.

    Parameters
    ----------
    dim: dimension of the sphere (embedded in R^{dim+1})
    theta: radius of the bubble distribution

    Returns
    -------
    tuple (modulation factor, std-dev on the modulation factor)
    """
    gamma = 1.0 / dim + (1.0 - 1.0 / dim) * theta / np.tan(theta)
    return (1.0 / gamma) ** 2


def plot_modulation_factor(n_sample, dim, n_expectation=1000, n_theta=20):
    """Plot the modulation factor curve w.r.t. the dispersion.

    Plot the curve of modulation factor on the convergence of the
    empirical Fréchet mean as a function of the radius of the bubble
    distribution and for n_sample points on the sphere S_dim
    embedded in R^{dim+1}.

    Parameters
    ----------
    n_sample: number of samples to draw
    dim: dimension of the sphere (embedded in R^{dim+1})
    n_expectation: number of computations for approximating the expectation
    n_theta: number of sampled radii for the bubble distribution

    Returns
    -------
    matplolib figure
    """
    theta = np.linspace(0.000001, np.pi / 2.0 - 0.000001, n_theta)
    measured_modulation_factor = []
    error = []
    small_var_modulation_factor = []
    asymptotic_modulation_factor = []
    for theta_i in theta:
        (var, std_var) = modulation_factor(
            n_sample, theta_i, dim, n_expectation=n_expectation)
        measured_modulation_factor.append(var)
        error.append(std_var)
        print(n_sample, theta_i, var, std_var, '\n')
        small_var_modulation_factor.append(
            1.0 + 2.0 / 3.0 * theta_i ** 2
            * (1.0 - 1.0 / dim) * (1.0 - 1.0 / n_sample))
        asymptotic_modulation_factor.append(
            asymptotic_modulation(dim, theta_i))
    plt.figure()
    plt.errorbar(theta, measured_modulation_factor,
                 yerr=error, color='r', label='Measured')
    plt.plot(theta, small_var_modulation_factor,
             'g', label='Small variance prediction')
    plt.plot(theta, asymptotic_modulation_factor,
             'grey', label='Asymptotic prediction')
    plt.xlabel(r'Standard deviation $\theta$')
    plt.ylabel(r'Modulation factor $\alpha$')
    plt.title("Convergence rate modulation factor, "
              "sphere dim={1}, n={0}".format(n_sample, dim))
    plt.legend(loc='best')
    plt.draw()
    plt.pause(0.01)
    return plt


def multi_plot_modulation_factor(dim, n_expectation=1000, n_theta=20):
    """Plot modulation factor curves for large number of samples.

    Plot several curves of modulation factor on the convergence of the
    empirical Fréchet mean as a function of the radius of the bubble
    distribution and for 10 to 100 sample points on the sphere S_dim
    embedded in R^{dim+1}.

    Parameters
    ----------
    dim: dimension of the sphere (embedded in R^{dim+1})
    n_expectation: number of computations for approximating the expectation
    n_theta: number of sampled radii for the bubble distribution

    Returns
    -------
    matplolib figure
    """
    theta = np.linspace(0.000001, np.pi / 2.0 - 0.000001, n_theta)
    small_var_modulation_factor = []
    asymptotic_modulation_actor = []
    plt.figure()
    for theta_i in theta:
        small_var_modulation_factor.append(
            1.0 + 2.0 / 3.0 * theta_i ** 2 * (1.0 - 1.0 / dim) * 1.0)
        asymptotic_modulation_actor.append(
            asymptotic_modulation(dim, theta_i))
    plt.plot(theta, small_var_modulation_factor,
             'g', label='Small variance prediction')
    plt.plot(theta, asymptotic_modulation_actor,
             'grey', label='Asymptotic prediction')
    color = {10: 'red', 20: 'orange', 50: 'olive', 100: 'blue'}
    for n_sample in [10, 20, 50, 100]:
        measured_modulation_factor = []
        for theta_i in theta:
            (var, std_var) = modulation_factor(
                n_sample, theta_i, dim, n_expectation=n_expectation)
            measured_modulation_factor.append(var)
            print(n_sample, theta_i, var, std_var, '\n')
        plt.plot(theta, measured_modulation_factor,
                 color=color[n_sample], label="N={0}".format(n_sample))
    plt.xlabel(r'Standard deviation $\theta$')
    plt.ylabel(r'Modulation factor $\alpha$')
    plt.legend(loc='best')
    plt.title("Convergence rate modulation factor, "
              "sphere, dim={0}, N > 5".format(dim))
    plt.draw()
    plt.pause(0.01)
    return plt


def main():
    """Visualise the uncertainty of the empirical Fréchet mean on the sphere.

    The variance of the Fréchet mean FM_n of a sample of n IID random variables
    of variance Var is decreasing more slowly in a sphere than in a Euclidean
    space. This example computes the  modulation factor
         alpha = Var( FM_n) / ( n * Var)
    for isotropic distributions on hyper-spheres of radius 0 < theta < Pi in
    the sphere S_dim (called here a bubble).
    """
    n_expect = 10

    print("Var of empirical mean for 1 sample, theta=0.1 in S2",
          empirical_frechet_var_bubble(
              1, 0.1, 2, n_expectation=n_expect), "\n")
    print("Var of empirical mean for 1 sample, theta=0.1 in S3",
          empirical_frechet_var_bubble(
              1, 0.1, 3, n_expectation=n_expect), "\n")

    print("Modulation factor for 1 sample theta=0.1 in S2 "
          "(should be close to 1):",
          modulation_factor(
              1, 0.1, 2, n_expectation=n_expect), "\n")

    print("Modulation factor for 500 sample theta close to Pi/2 in S5 "
          "(should be around 25):",
          modulation_factor(
              500, gs.pi / 2 - 0.001, 5, n_expectation=n_expect), "\n")

    plot_modulation_factor(2, 2, n_expectation=n_expect)
    plot_modulation_factor(4, 2, n_expectation=n_expect)
    plot_modulation_factor(6, 2, n_expectation=n_expect)

    multi_plot_modulation_factor(2, n_expectation=n_expect)

    plot_modulation_factor(2, 3, n_expectation=n_expect)
    plot_modulation_factor(4, 3, n_expectation=n_expect)
    plot_modulation_factor(6, 3, n_expectation=n_expect)

    multi_plot_modulation_factor(3, n_expectation=n_expect)

    plot_modulation_factor(2, 4, n_expectation=n_expect)
    plot_modulation_factor(4, 4, n_expectation=n_expect)
    plot_modulation_factor(6, 4, n_expectation=n_expect)

    multi_plot_modulation_factor(4, n_expectation=n_expect)

    plt.figure()
    plt.show()


if __name__ == "__main__":
    main()
