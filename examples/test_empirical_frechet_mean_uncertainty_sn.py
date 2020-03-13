"""Visualise the uncertainty of the empirical Fréchet mean on the sphere.

The variance of the Fréchet mean FM_n of a sample of n IID
random variables of variance Var is decreasing more slowly
in a sphere than in a Euclidean space.
This example computes the  modulation factor
     alpha = Var( FM_n) / ( n * Var)
for isotropic distributions on hyperspheres or radius 0<r<Pi
in the sphere s3.
"""

import matplotlib.pyplot as plt
import numpy as np

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere


def my_random_uniform_sphere(dim):
    """Compute one random uniform sample on the sphere.

    The method is to draw uniformly in a hypercube
    and to cut whatever is out of the unit sphere
    or too small for renormalization.

    Parameters
    ----------
    dim: dimension of the sphere (embedded in R^{dim+1})

    Returns
    -------
    array-like, shape=[dimension]
    """
    vector = (np.random.random_sample(dim + 1) * 2.0 - 1.0)
    norm = np.linalg.norm(vector)
    while norm > 1 or norm < 1.e-10:
        vector = (np.random.random_sample(dim + 1) * 2.0 - 1.0)
        norm = np.linalg.norm(vector)
    return vector / norm


def my_random_uniform_sample_hypersphere(n_sample, dim):
    """Compute n_sample random uniform sample on the sphere.

    The method is to draw uniformly in a hypercube
    and to cut whatever is out of the unit sphere
    or too small for renormalization.

    Parameters
    ----------
    n_sample: number of samples to draw
    dim: dimension of the sphere (embedded in R^{dim+1})

    Returns
    -------
    array-like, shape=[n_sample, dimension]
    """
    data = np.zeros((n_sample, dim + 1), dtype=gs.float64)
    print("coucou\n")
    for i in range(n_sample):
        data[i, :] = my_random_uniform_sphere(dim)
    return data


def empirical_frechet_var_shell(n_sample, theta, dim,
                                n_expectation=1000):
    """Variance of the empirical Fréchet mean for a shell distribution.

    Draw n_samples from a shell distribution, computes its empirical
    Fréchet mean and the square distance to the asymptotic mean. This
    is repeated n_expectation times to compute an approximation of its
    expectation (i.e. its variance) by sampling.

    The shell distribution is an isotropic distributions on a Riemannian
    hyper sub-sphere of radius 0 < theta < Pi around the north pole of the
    sphere of dimension dim.

    Parameters
    ----------
    n_sample: number of samples to draw
    theta: radius of the shell distribution
    dim: dimension of the sphere (embedded in R^{dim+1})
    n_expectation: number of computations for approximating the expectation

    Returns
    -------
    tuple (variance, std-dev on the computed variance)
    """
    assert dim > 1, "Dim > 1 needed to draw a uniform sample on sub-sphere"
    var = []
    sphere = Hypersphere(dimension=dim)
    shell = Hypersphere(dimension=dim - 1)

    # Define north pole
    north_pole = np.zeros(dim + 1)
    north_pole[dim] = 1.0
    for k in range(n_expectation):
        # Sample n points from the uniform distribution on a sub-sphere
        # of radius theta (i.e cos(theta) in ambient space)
        data = gs.zeros((n_sample, dim + 1), dtype=gs.float64)
        # For sampling on a sub-sphere, use RandomUniform(dim-1)
        # directions = shell.random_uniform(n_sample)
        # Alternative sampling
        directions = my_random_uniform_sample_hypersphere(n_sample, dim-1)
        for i in range(n_sample):
            for j in range(dim):
                data[i, j] = gs.sin(theta) * directions[i, j]
            data[i, dim] = gs.cos(theta)
        # Compute empirical Fréchet mean of the n-sample
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
    theta: radius of the shell distribution
    dim: dimension of the sphere (embedded in R^{dim+1})
    n_expectation: number of computations for approximating the expectation

    Returns
    -------
    tuple (modulation factor, std-dev on the modulation factor)
    """
    (var, std_var) = empirical_frechet_var_shell(
        n_sample, theta, dim, n_expectation=n_expectation)
    return var * n_sample / theta ** 2, std_var * n_sample / theta ** 2


def asymptotic_modulation(dim, theta):
    """Compute the asymptotic modulation factor.

    Parameters
    ----------
    dim: dimension of the sphere (embedded in R^{dim+1})
    theta: radius of the shell distribution

    Returns
    -------
    tuple (modulation factor, std-dev on the modulation factor)
    """
    gamma = 1.0 / dim + (1.0 - 1.0 / dim) * theta / np.tan(theta)
    return (1.0 / gamma) ** 2


def plot_modulation_factor(n_sample, dim, n_expectation=1000, n_theta=20):
    """Plot the modulation factor curve w.r.t. the dispersion.

    Plot the modulation factor on the convergence of the empirical
    Fréchet mean for different radii of the shell distribution.

    Parameters
    ----------
    n_sample: number of samples to draw
    dim: dimension of the sphere (embedded in R^{dim+1})
    n_expectation: number of computations for approximating the expectation
    n_theta: number of sampled radii for the shell distribution

    Returns
    -------
    matplolib figure
    """
    # modulation factor for n points on the sphere S_dim embedded in R^{dim+1}
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
    # plt.savefig("Figures/SphVarModulation_N{0}_d{1}.svg".format(n, dim))
    # plt.savefig("Figures/SphVarModulation_N{0}_d{1}.pdf".format(n, dim))
    return plt


def multi_plot_modulation_factor(dim, n_expectation=1000, n_theta=20):
    """Plot modulation factor curves for large number of samples.

    Parameters
    ----------
    dim: dimension of the sphere (embedded in R^{dim+1})
    n_expectation: number of computations for approximating the expectation
    n_theta: number of sampled radii for the shell distribution

    Returns
    -------
    matplolib figure
    """
    # Implementation for the sphere S_dim in R^{dim+1}
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
    # plt.savefig("Figures/SphVarModulation_N10p_d{0}.svg".format(dim))
    # plt.savefig("Figures/SphVarModulation_N10p_d{0}.pdf".format(dim))
    return plt


def main():
    """Visualise the uncertainty of the empirical Fréchet mean on the sphere.

    The variance of the Fréchet mean FM_n of a sample of n IID
    random variables of variance Var is decreasing more slowly
    in a sphere than in a Euclidean space.
    This example computes the  modulation factor
         alpha = Var( FM_n) / ( n * Var)
    for isotropic distributions on hyperspheres or radius 0<r<Pi
    in the sphere s3.
    """
    n_expect = 10

    print("Var of empirical mean for 1 sample, theta=0.1 in S2",
          empirical_frechet_var_shell(
              1, 0.1, 2, n_expectation=n_expect), "\n")
    print("Var of empirical mean for 1 sample, theta=0.1 in S3",
          empirical_frechet_var_shell(
              1, 0.1, 3, n_expectation=n_expect), "\n")

    ## Computation time problem:
    # time empirical_frechet_var_shell(3, 0.5, 4, n_expectation=1000)
    # -> 47 sec against 1.3 s for my previous implementation
    # time EmpiricalVar(3, 0.5, 4, NN=1000)
    # but random sampling does not seem to be the problem






    print("Modulation factor for 1 sample theta=0.1 in S2 "
          "(should be close to 1):",
          modulation_factor(
              1, 0.1, 2, n_expectation=n_expect), "\n")

    print("Modulation factor for 500 sample theta close to Pi/2 in S5 "
          "(should be around 25):",
          modulation_factor(
              500, gs.pi / 2 - 0.001, 5, n_expectation=n_expect), "\n")

    plot_modulation_factor(2, 2, n_expectation=n_expect)
    # plot_modulation_factor(3, 2, n_expectation=n_expect)
    plot_modulation_factor(4, 2, n_expectation=n_expect)
    # plot_modulation_factor(5, 2, n_expectation=n_expect)
    plot_modulation_factor(10, 2, n_expectation=n_expect)

    # plot_modulation_factor(1, 3, n_expectation=n_expect)
    plot_modulation_factor(2, 3, n_expectation=n_expect)
    # plot_modulation_factor(3, 3, n_expectation=n_expect)
    plot_modulation_factor(4, 3, n_expectation=n_expect)
    # plot_modulation_factor(5, 3, n_expectation=n_expect)

    multi_plot_modulation_factor(3, n_expectation=n_expect)

    multi_plot_modulation_factor(4, n_expectation=n_expect)

    plt.figure()
    plt.show()


if __name__ == "__main__":
    main()
