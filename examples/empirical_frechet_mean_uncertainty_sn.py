"""Visualise the uncertainty of the empirical Fréchet mean on the sphere.

The variance of the Fréchet mean FM_n of a sample of n IID random variables
of variance Var is decreasing more slowly in a sphere than in a Euclidean
space. This example computes the  modulation factor
     alpha = Var( FM_n) / ( n * Var)
for isotropic distributions on hyper-spheres of radius 0 < theta < Pi in
the sphere S_dim (called here a bubble).
"""

import logging

import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import FrechetMean


def empirical_frechet_var_bubble(n_samples, theta, dim, n_expectation=1000):
    """Variance of the empirical Fréchet mean for a bubble distribution.

    Draw n_sampless from a bubble distribution, computes its empirical
    Fréchet mean and the square distance to the asymptotic mean. This
    is repeated n_expectation times to compute an approximation of its
    expectation (i.e. its variance) by sampling.

    The bubble distribution is an isotropic distributions on a Riemannian
    hyper sub-sphere of radius 0 < theta < Pi around the north pole of the
    sphere of dimension dim.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    theta: float
        Radius of the bubble distribution.
    dim : int
        Dimension of the sphere (embedded in R^{dim+1}).
    n_expectation: int, optional (defaults to 1000)
        Number of computations for approximating the expectation.

    Returns
    -------
    tuple (variance, std-dev on the computed variance)
    """
    if dim <= 1:
        raise ValueError("Dim > 1 needed to draw a uniform sample on sub-sphere.")
    var = []
    sphere = Hypersphere(dim=dim)
    bubble = Hypersphere(dim=dim - 1)

    north_pole = gs.zeros(dim + 1)
    north_pole[dim] = 1.0
    for _ in range(n_expectation):
        # Sample n points from the uniform distribution on a sub-sphere
        # of radius theta (i.e cos(theta) in ambient space)
        # TODO (nina): Add this code as a method of hypersphere
        last_col = gs.cos(theta) * gs.ones(n_samples)
        last_col = last_col[:, None] if (n_samples > 1) else last_col

        directions = bubble.random_uniform(n_samples)
        rest_col = gs.sin(theta) * directions
        data = gs.concatenate([rest_col, last_col], axis=-1)

        estimator = FrechetMean(
            sphere.metric, max_iter=32, method="adaptive", init_point=north_pole
        )
        if n_samples == 1:
            data = gs.expand_dims(data, 0)

        estimator.fit(data)
        current_mean = estimator.estimate_
        var.append(sphere.metric.squared_dist(north_pole, current_mean))
    return gs.mean(var), 2 * gs.std(var) / gs.sqrt(n_expectation)


def modulation_factor(n_samples, theta, dim, n_expectation=1000):
    """Modulation factor on the convergence of the empirical Fréchet mean.

    The modulation factor is the ratio of the variance of the empirical
    Fréchet mean on the manifold to the variance in a Euclidean space,
    for n_sampless drawn from an isotropic distributions on a Riemannian
    hyper sub-sphere of radius 0 < theta < Pi around the north pole of the
    sphere of dimension dim.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    theta : float
        Radius of the bubble distribution.
    dim : int
        Dimension of the sphere (embedded in R^{dim+1}).
    n_expectation: int, optional (defaults to 1000)
        Number of computations for approximating the expectation.

    Returns
    -------
    tuple (modulation factor, std-dev on the modulation factor)
    """
    (var, std_var) = empirical_frechet_var_bubble(
        n_samples, theta, dim, n_expectation=n_expectation
    )
    return var * n_samples / theta**2, std_var * n_samples / theta**2


def asymptotic_modulation(dim, theta):
    """Compute the asymptotic modulation factor.

    Parameters
    ----------
    dim : int
        Dimension of the sphere (embedded in R^{dim+1}).
    theta : float
        Radius of the bubble distribution.

    Returns
    -------
    tuple (modulation factor, std-dev on the modulation factor)
    """
    gamma = 1.0 / dim + (1.0 - 1.0 / dim) * theta / gs.tan(theta)
    return (1.0 / gamma) ** 2


def plot_modulation_factor(n_samples, dim, n_expectation=1000, n_theta=20):
    """Plot the modulation factor curve w.r.t. the dispersion.

    Plot the curve of modulation factor on the convergence of the
    empirical Fréchet mean as a function of the radius of the bubble
    distribution and for n_samples points on the sphere S_dim
    embedded in R^{dim+1}.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw
    dim : int
        Dimension of the sphere (embedded in R^{dim+1}).
    n_expectation: int, optional (defaults to 1000)
        Number of computations for approximating the expectation.
    n_theta: int, optional (defaults to 20)
        Number of sampled radii for the bubble distribution.

    Returns
    -------
    matplolib figure
    """
    theta = gs.linspace(0.000001, gs.pi / 2.0 - 0.000001, n_theta)
    measured_modulation_factor = []
    error = []
    small_var_modulation_factor = []
    asymptotic_modulation_factor = []
    for theta_i in theta:
        (var, std_var) = modulation_factor(
            n_samples, theta_i, dim, n_expectation=n_expectation
        )
        measured_modulation_factor.append(var)
        error.append(std_var)
        logging.info("{} {} {} {}\n".format(n_samples, theta_i, var, std_var))
        small_var_modulation_factor.append(
            1.0 + 2.0 / 3.0 * theta_i**2 * (1.0 - 1.0 / dim) * (1.0 - 1.0 / n_samples)
        )
        asymptotic_modulation_factor.append(asymptotic_modulation(dim, theta_i))
    plt.figure()
    plt.errorbar(
        theta, measured_modulation_factor, yerr=error, color="r", label="Measured"
    )
    plt.plot(theta, small_var_modulation_factor, "g", label="Small variance prediction")
    plt.plot(theta, asymptotic_modulation_factor, "grey", label="Asymptotic prediction")
    plt.xlabel(r"Standard deviation $\theta$")
    plt.ylabel(r"Modulation factor $\alpha$")
    plt.title("Convergence rate modulation factor, " f"sphere dim={n_samples}, n={dim}")
    plt.legend(loc="best")
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
    n_expectation = 10

    logging.info(
        "Var of empirical mean for 1 sample, theta=0.1 "
        "in S2 {} \n".format(
            empirical_frechet_var_bubble(1, 0.1, 2, n_expectation=n_expectation)
        )
    )
    logging.info(
        "Var of empirical mean for 1 sample, theta=0.1 "
        "in S3 {} \n".format(
            empirical_frechet_var_bubble(1, 0.1, 3, n_expectation=n_expectation)
        )
    )

    logging.info(
        "Modulation factor for 1 sample theta=0.1 in S2 "
        "(should be close to 1): {} \n".format(
            modulation_factor(1, 0.1, 2, n_expectation=n_expectation)
        )
    )

    logging.info(
        "Modulation factor for 500 sample theta close to Pi/2 in S5 "
        "(should be around 25): {} \n".format(
            modulation_factor(500, gs.pi / 2 - 0.001, 5, n_expectation=n_expectation)
        )
    )

    plot_modulation_factor(2, 2, n_expectation=n_expectation)

    plt.figure()
    plt.show()


if __name__ == "__main__":
    main()
