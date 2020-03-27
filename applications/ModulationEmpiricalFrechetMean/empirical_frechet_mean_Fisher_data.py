"""Modulation of empirical Fréchet mean on spherical/projective Fisher data.

The variance of the Fréchet mean FM_n of a sample of n IID random variables
of variance Var is decreasing more slowly in a sphere than in a Euclidean
space. This example computes the  modulation factor
     alpha = Var( FM_n) / ( n * Var)
for bootstrap samples from a few real-world empirical distributions on the
sphere S2 and the projective space P2 taken from the book of Fisher, Lewis and
Embleton 1987.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import random
from xlrd import open_workbook
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import _adaptive_gradient_descent


def mean_sq_dist_s2(location, data):
    """compute the mean-square deviation from location to data"

    Parameters
    ----------
    data: empirical distribution
    location: location where to compute MSD

    Returns
    -------
    MSD: mean square deviation
    """
    n_samples, dim = gs.shape(data)
    assert n_samples > 0, "Dataset needs to have at least one data"
    assert dim > 1, "Embedding dimension must be at least 2 (was " \
                    "{0})".format(dim)
    sphere = Hypersphere(dim - 1)

    msd = 0.0
    for item in data:
        sq_dist = sphere.metric.squared_dist(location, item)[0, 0]
        msd = msd + sq_dist
    return msd / n_samples


def msd_hbar_s2(location, data):
    """compute the mean-square deviation from the location to the points of
    the dataset and the mean Hessian of square distance at the location"""
    sphere = Hypersphere(2)
    num_sample = len(data)
    assert num_sample > 0, "Dataset needs to have at least one data"
    var = 0.0
    hbar = 0.0
    for item in data:
        sq_dist = sphere.metric.squared_dist(location, item)[0, 0]
        var = var + sq_dist
        # hbar = E(h(dist ^ 2)) with h(t) = sqrt(t) cot( sqrt(t) )  for kappa=1
        if sq_dist > 1e-4:
            d = gs.sqrt(sq_dist)
            h = d / gs.tan(d)
        else:
            h = 1.0 - sq_dist / 3.0 - sq_dist ** 2 / 45.0 - 2 / 945 * \
                sq_dist ** 3 - sq_dist ** 4 / 4725
        hbar = hbar + h
    return var / num_sample, hbar / num_sample


def cov_hessian_covmean_sphere(mean, dataset):
    """compute dataset covariance, Hessian and covariance of the Fréchet mean"

    Returns
    -------
    cov: Cov = 1/n \\sum_i=1^n log_mean(x_i) log_mean(x_i)^t
    hess: H = 1/n \\sum_i=1^n \\partial^2 \\dist^2(mean, x_i) / \\partial mean^2
    hinv: inverse Hessian restricted to the tangent space
    cov_mean_clt: covariance predicted on mean by CLT
    cov_mean_hc: covariance predicted on mean by high concentration expansion
    """
    n_samples, dim = gs.shape(dataset)
    assert n_samples > 0, "dataset needs to have at least one data"
    assert dim > 1, "Dimension of embedding space should be at least 2"
    assert len(mean) == dim, "Mean should have dimension dim"

    sphere = Hypersphere(dim - 1)
    cov = np.zeros((dim, dim), 'float')
    hess = np.zeros((dim, dim), 'float')
    idmat = np.identity(dim, 'float')
    for item in dataset:
        xy = sphere.metric.log(item, mean)
        theta = sphere.metric.norm(xy, mean)
        cov += np.outer(xy, xy)
        # Numerical issues
        if theta < 1e-3:
            a = 1. / 3. + theta ** 2 / 45. + (2. / 945.) * theta ** 4 \
                + theta ** 6 / 4725 + (2. / 93555.) * theta ** 8
            b = 1. - theta ** 2 / 3. - theta ** 4 / 45. \
                - (2. / 945.) * theta ** 6 - theta ** 8 / 4725
        else:
            a = 1. / theta ** 2 - 1.0 / np.tan(theta) / theta
            b = theta / np.tan(theta)
        hess += a * np.outer(xy, xy) + b * (idmat - np.outer(mean, mean))
    hess = 2 * hess / n_samples
    cov = cov / n_samples
    # ATTN inverse should be restricted to the tangent space because mean is
    # an eigenvector of Hessian with eigenvalue 0
    # Thus, we add a component along mean.mean^t to make eigenvalue 1 before
    # inverting
    hinv = np.linalg.inv(hess + np.outer(mean, mean))
    # and we remove the component along mean.mean^t afterward
    hinv -= np.dot(mean, np.matmul(hinv, mean)) * np.outer(mean, mean)
    # Compute covariance of the mean predicted
    cov_mean_clt = 4 * np.matmul(hinv, np.matmul(cov, hinv)) / n_samples
    cov_mean_hc = np.matmul(cov, 2.0 * (idmat - (1.0 - 1.0 / n_samples) / 3.0 *
                                        (cov - np.trace(
                                            cov) * idmat))) / n_samples
    return cov, hess, hinv, cov_mean_clt, cov_mean_hc


def isotropic_modulation_factor_sphere(mean, data):
    """compute isotropic modulation factor for Fréchet mean on sphere"

     Parameters
     ----------
     data: empirical distribution to bootstrap from
     mean: Fréchet mean of that distribution

     Returns
     -------
     alpha_clt: modulation predicted by CLT
     alpha_hc: modulation predicted by high concentration expansion
     """
    n_samples = len(data)
    assert n_samples > 0, "dataset needs to have at least one data"
    dim = len(data[0]) - 1
    assert dim > 0, "sphere dimension needs to be at least 1"
    var, hbar = msd_hbar_s2(mean, data)
    alpha_clt = (1.0 / dim + (1.0 - 1.0 / dim) * hbar) ** (-2)
    alpha_hc = 1.0 + 2.0 / 3.0 * var * (1.0 - 1.0 / dim) * (1.0 - 1.0
                                                            / n_samples)
    return alpha_clt, alpha_hc


def anisotropic_modulation_factor_sphere(mean, data):
    """compute anisotropic modulation factor for Fréchet mean on sphere"

    Parameters
    ----------
    data: empirical distribution to bootstrap from
    mean: Fréchet mean of that distribution

    Returns
    -------
    alpha_clt: modulation predicted by CLT
    alpha_hc: modulation predicted by high concentration expansion
    """
    mean = gs.to_ndarray(mean, to_ndim=2)
    (cov, hess, hinv, cov_mean_clt, cov_mean_hc) = \
        cov_hessian_covmean_sphere(mean, data)
    sig2 = np.trace(cov)
    sig2_hc = np.trace(cov_mean_hc)
    sig2_clt = np.trace(cov_mean_clt)
    print("Var = {0}, VarMeanSM = {1}, VarMeanCLT = {2}".format(sig2, sig2_hc,
                                                                sig2_clt))
    alpha_hc = sig2_hc / sig2 * len(data)
    alpha_clt = sig2_clt / sig2 * len(data)
    print(
        "Anisotropic modulation factor: small var {0} / asymptotic {1}".format(
            alpha_hc, alpha_clt))
    return alpha_hc, alpha_clt


def max_extent_s2(location, data):
    """compute maximal extension of the dataset in two orthogonal directions"""
    sphere = Hypersphere(2)
    num_sample = len(data)
    assert num_sample > 0, "Dataset needs to have at least one data point"
    max_1 = 0.0
    log_1 = sphere.metric.log(location, location)  # point, base_point
    for item in data:
        xy = sphere.metric.log(item, location)
        theta = sphere.metric.norm(xy, location)[0, 0]
        if theta > max_1:
            max_1 = theta
            log_1 = xy
    log_1 = log_1 / sphere.metric.norm(log_1, location)[0, 0]
    max_2 = 0.0
    for item in data:
        xy = sphere.metric.log(item, location)
        # project to get the orthogonal part to Log_1
        xy = xy - sphere.metric.inner_product(xy, log_1, location) * log_1
        theta = sphere.metric.norm(xy, location)[0, 0]
        if theta > max_2:
            max_2 = theta
    return max_1, max_2


def stat_dataset_s2(data, title, frechet_mean):
    """Print concentration statistics on a dataset on the sphere S2"

    Parameters
    ----------
    data: spherical samples to be drawn
    title: title of dataset
    frechet_mean: list of Fréchet mean points
    """
    assert len(frechet_mean) > 0, "Fréchet mean or reference point needs to " \
                                  "be provided"

    # print some statistics on the concentration of the dataset
    var, hbar = msd_hbar_s2(frechet_mean[0], data)
    print("{2}: Var {0} rad (Stddev {1} deg)".format(var, np.sqrt(
        var) * 180 / gs.pi, title))
    (max1, max2) = max_extent_s2(frechet_mean[0], data)
    print("{2}: Extent {0} deg / {1} deg".format(max1 * 180 / gs.pi,
                                                 max2 * 180 / gs.pi,
                                                 title))


def plot_dataset_s2(dataset, title, frechet_mean_list=[]):
    """Plot a dataset on the sphere S2 with it mean in red"

    Parameters
    ----------
    dataset: spherical samples to be drawn
    title: title of the plot
    frechet_mean_list: list of Frechet mean points

    Returns
    -------
    matplotlib plot
    """
    # create sphere
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Visualisation of the dataset as a point set
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='c', alpha=0.3,
                    linewidth=0)
    for unit_vect in dataset:
        ax.scatter(unit_vect[0], unit_vect[1], unit_vect[2], c='b', marker='.')
    ax.text2D(0.5, 1, title, transform=ax.transAxes)
    if len(frechet_mean_list) > 0:
        for unit_vect in frechet_mean_list:
            ax.scatter(unit_vect[0], unit_vect[1], unit_vect[2], c='r',
                       marker='o')
    # plt.show()
    plt.draw()  # non blocking in a script
    plt.pause(0.01)
    return plt


def polar_2_unit_axis(colatitude, long):
    """Polar coordinates to unit axis parametrisation of S2"

    Parameters
    ----------
    colatitude theta (angle from north pole) in degrees
    Longitude phi in degrees

    Returns
    -------
    3D unit vector
    """
    theta = colatitude * gs.pi / 180.0
    phi = long * gs.pi / 180.0
    return np.array(
        [gs.sin(theta) * gs.cos(phi), gs.sin(theta) * gs.sin(phi),
         gs.cos(theta)])


def geographical_2_unit_axis(latitude, longitude):
    """Geographical coordinates to unit axis parametrisation of S2"

    Parameters
    ----------
    latitude theta (angle from equator, positive towards north) in degrees
    Longitude phi in degrees

    Returns
    -------
    3D unit vector
    """
    return polar_2_unit_axis(90.0 - latitude, longitude)


def geological_2_unit_axis(dip, dip_direction):
    """Geological coordinates to unit axis parametrisation of S2"

    Parameters
    ----------
    dip in degrees
    dip_direction in degrees

    Returns
    -------
    3D unit vector
    """
    return polar_2_unit_axis(dip, 360.0 - dip_direction)


def empirical_frechet_mean_random_init_s2(data, n_init=1, init_points=[]):
    """Fréchet mean on S2 by gradient descent from multiple starting points"

    Parameters
    ----------
    data: empirical distribution on S2
    n_init: number of initial points drawn uniformly at random on S2
    init_points: list of initial points for the first gradient descent

    Returns
    -------
    frechet mean list
    """
    assert n_init >= 1, "Gradient descent needs at least one starting point"
    dim = len(data[0]) - 1
    sphere = Hypersphere(dimension=dim)
    if len(init_points) == 0:
        init_points = [sphere.random_uniform()]
        # for a noncompact manifold, we need to revise this to a ball
        # with a maximal radius

    mean = _adaptive_gradient_descent(data,
                                      metric=sphere.metric,
                                      n_max_iterations=64,
                                      init_points=init_points)

    sigma_mean = mean_sq_dist_s2(mean, data)
    # print ("variance {0} for FM {1}".format(sigFM,FM))
    for i in range(n_init - 1):
        init_points = sphere.random_uniform()
        new_mean = _adaptive_gradient_descent(data,
                                              metric=sphere.metric,
                                              n_max_iterations=64,
                                              init_points=init_points)
        sigma_new_mean = mean_sq_dist_s2(new_mean, data)
        if sigma_new_mean < sigma_mean:
            mean = new_mean
            sigma_mean = sigma_new_mean
            # print ("new variance {0} for FM {1}".format(sigFM,FM))
    return mean


def empirical_frechet_var_bootstrap_s2(data, mean, n_samples, n_init,
                                       n_expectation=1000):
    """Variance of the empirical Fréchet mean for a bootstrap distribution.

    Draw n_samples from an empirical distribution, computes its empirical
    Fréchet mean and the square distance to the asymptotic mean. This
    is repeated n_expectation times to compute an approximation of its
    expectation (i.e. its variance) by sampling.

    Parameters
    ----------
    data: empirical distribution to bootstrap from
    mean: Fréchet mean of that distribution
    n_samples: number of samples to draw
    n_init: number of initial points for gradient descent
    n_expectation: number of computations for approximating the expectation

    Returns
    -------
    tuple (variance, std-dev on the computed variance)
    """
    assert n_init >= 1, "Gradient descent needs at least one starting point"
    dim = len(data[0]) - 1
    sphere = Hypersphere(dimension=dim)
    sq_dist = []
    for i in range(n_expectation):
        # bootstrap n_samples points from the empirical distribution data
        bootstrap = []
        for j in range(n_samples):
            bootstrap.append(random.choice(data))
        mean_i = empirical_frechet_mean_random_init_s2(bootstrap,
                                                       n_init=n_init,
                                                       init_points=[mean])
        sq_dist.append(sphere.metric.squared_dist(mean_i, mean)[0, 0])
    return np.mean(sq_dist), np.std(sq_dist) / np.sqrt(n_expectation - 1.0)


# def empirical_frechet_modulation_bootstrap_s2(data, mean, n_samples, n_init,
#                                               n_expectation=1000):
#     """Modulation of the empirical Fréchet mean for a bootstrap distribution.
#
#     The modulation factor is the ratio of the variance of the empirical
#     Fréchet mean on the manifold to the variance in a Euclidean space,
#     for a bootstrap n_sampless drawn from an empirical distributions data on a
#     Riemannian sphere of dimension 2.
#
#     Parameters
#     ----------
#     data: empirical distribution to bootstrap from
#     mean: Fréchet mean of that distribution
#     n_samples: number of samples to draw
#     n_init: number of initial points for gradient descent
#     n_expectation: number of computations for approximating the expectation
#
#     Returns
#     -------
#     tuple (variance, std-dev on the modulation factor)
#     """
#     var_mean, var_stddev = \
#         empirical_frechet_var_bootstrap_s2(data, mean, n_samples,
#                                            n_init, n_expectation)
#     var = mean_sq_dist_s2(mean, data)
#     return var_mean * n_samples / var, var_stddev * n_samples / var


def empirical_frechet_modulation_bootstrap_s2(data, mean,
                                              n_init=1,
                                              n_expectation=1000,
                                              n_samples_list=[2, 100]):
    """Modulation of the empirical Fréchet mean for a bootstrap distribution.

    The modulation factor is the ratio of the variance of the empirical
    Fréchet mean on the manifold to the variance in a Euclidean space,
    for a bootstrap n_samples drawn from an empirical distributions data
    on a Riemannian sphere of dimension 2.

    This function computes the observed and predicted modulation factor for
    n_expectation bootstrap samples of size n_sample for each n_sample from
    the n_samples_list.  It returns lists that can be plotted independently.

    Parameters
    ----------
    data: empirical distribution to bootstrap from
    mean: Fréchet mean of that distribution
    n_init: number of initial points for gradient descent
    n_expectation: number of computations for approximating the expectation
    n_samples_list: list of number of samples

    Returns
    -------
    alpha_mean: mean measured modulation list
    alpha_std: std dev on the mean measured modulation list
    alpha_iso_hc: predicted isotropic high concentration modulation list
    alpha_aniso_hc: predicted anisotropic high concentration modulation list
    alpha_iso_clt: : predicted isotropic asymptotic CLT modulation list
    alpha_aniso_clt: : predicted aniisotropic asymptotic CLT modulation list
    """

    # compute isotropic modulation factor for Fréchet mean on sphere
    n_samples_orig, dim = gs.shape(data)
    assert n_samples_orig > 0, "dataset needs to have at least one data"
    dim = dim - 1
    assert dim > 0, "sphere dimension needs to be at least one"

    assert len(mean) == 1, "Mean should be a list with a unique mean"
    mean = mean[0]
    assert len(mean) == dim + 1, "Mean should have same dimension as data"

    var, hbar = msd_hbar_s2(mean, data)
    # alpha_iso_clt_orig = (1.0 / dim + (1.0 - 1.0 / dim) * hbar) ** (-2)
    # alpha_iso_hc_orig = 1.0 + 2.0 / 3.0 * var \
    # * (1.0 - 1.0 / dim) * (1.0 - 1.0 / n_samples_orig)

    # compute anisotropic modulation factor for Fréchet mean on sphere
    (cov, hess, hinv, cov_mean_clt, cov_mean_hc) = \
        cov_hessian_covmean_sphere(mean, data)
    # sig2 = np.trace(cov)
    # sig2_hc = np.trace(cov_mean_hc)
    # sig2_clt = np.trace(cov_mean_clt)
    # aniso_hc_alpha_orig = sig2_hc / sig2 * n_samples_orig
    iso_clt_alpha = (1.0 / dim + (1.0 - 1.0 / dim) * hbar) ** (-2)
    aniso_clt_alpha = np.trace(cov_mean_clt) / var * n_samples_orig
    idmat = np.identity(dim + 1, 'float')
    kappa = +1.0  # sectional curvature of the sphere

    list_alpha_mean = []
    list_alpha_std = []
    list_alpha_iso_hc = []
    list_alpha_aniso_hc = []
    list_alpha_iso_clt = []
    list_alpha_aniso_clt = []

    for n_samples in n_samples_list:
        var_mean, var_stddev = \
            empirical_frechet_var_bootstrap_s2(data, mean, n_samples,
                                               n_init, n_expectation)
        mean_alpha = var_mean * n_samples / var
        std_alpha = var_stddev * n_samples / var
        list_alpha_mean.append(mean_alpha)
        list_alpha_std.append(std_alpha)
        print("{0} samples : Measured modulation = {1} pm {2}".format(
            n_samples, mean_alpha, std_alpha))

        # Asymptotic CLT predictions of modulation are independent of n_samples
        list_alpha_iso_clt.append(iso_clt_alpha)
        list_alpha_aniso_clt.append(aniso_clt_alpha)

        # Compute HC isotropic predictions of modulation
        iso_hc_alpha = 1.0 + 2.0 / 3.0 * var \
                       * (1.0 - 1.0 / dim) * (1.0 - 1.0 / n_samples)
        list_alpha_iso_hc.append(iso_hc_alpha)

        print("{0} samples : Predicted isotropic modulation = HC {1} / "
              "CLT {2}".format(n_samples, iso_hc_alpha, iso_clt_alpha))

        # Compute HC and CLT anisotropic predictions of modulation
        # aniso_hc_alpha needs is non linear in n_sample: recompute it
        aniso_cov = np.matmul(cov, (
                idmat - 2.0 * kappa * (1.0 - 1.0 / n_samples) / 3.0 * (
                cov - np.trace(cov) * idmat))) / n_samples
        aniso_hc_alpha = np.trace(aniso_cov) / var * n_samples
        list_alpha_aniso_hc.append(aniso_hc_alpha)

        print("{0} samples : Predicted anisotropic modulation = HC {1} / "
              "CLT {2}".format(n_samples, aniso_hc_alpha, aniso_clt_alpha))

    return list_alpha_mean, list_alpha_std, list_alpha_iso_hc, \
           list_alpha_aniso_hc, list_alpha_iso_clt, list_alpha_aniso_clt


def plot_empirical_frechet_modulation_bootstrap_s2(data, title, mean,
                                                   n_init=1,
                                                   n_expectation=1000):
    """Modulation of the empirical Fréchet mean for a bootstrap distribution.

    The modulation factor is the ratio of the variance of the empirical
    Fréchet mean on the manifold to the variance in a Euclidean space,
    for a bootstrap n_samples drawn from an empirical distributions data
    on a Riemannian sphere of dimension 2.

    This function computes the observed and predicted modulation factor for
    n_expectation bootstrap samples of size n_sample for each n_sample from
    the n_samples_list.  It returns lists that can be plotted independently.

    Parameters
    ----------
    data: empirical distribution to bootstrap from
    title: title of the plot
    mean: Fréchet mean of that distribution
    n_init: number of initial points for gradient descent
    n_expectation: number of computations for approximating the expectation

    Returns
    -------
    matplotlib figure
    """
    assert len(mean) == 1, "Mean should be a list with a unique mean"

    n_samples_list = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 30, 40, 50, 100, 200]
    alpha_mean, alpha_std, alpha_iso_hc, alpha_aniso_hc, alpha_iso_clt, \
    alpha_aniso_clt = empirical_frechet_modulation_bootstrap_s2(data, mean,
                                                                n_init,
                                                                n_expectation,
                                                                n_samples_list)

    plt.figure()
    ax = plt.axes()
    ax.set_xscale("log", nonposx='clip')
    # plt.loglog(NumSample, MeanAlpha, color='r', label='Measured')
    ax.errorbar(n_samples_list, alpha_mean, yerr=alpha_std, color='r',
                label='Measured')
    plt.plot(n_samples_list, alpha_iso_hc, 'g',
             label='Isotropic high concentration prediction',
             linestyle='--')
    plt.plot(n_samples_list, alpha_iso_clt, 'grey',
             label='Isotropic asymptotic CLT prediction',
             linestyle='--')
    plt.plot(n_samples_list, alpha_aniso_hc, 'g',
             label='Anisotropic high concentration prediction')
    plt.plot(n_samples_list, alpha_aniso_clt, 'grey',
             label='Anisotropic asymptotic CLT prediction')
    plt.xlabel(r'Number of samples')
    plt.ylabel(r'Modulation factor for boostrap distributions')
    plt.title("Modulation of convergence rate for spherical dataset {0}".format(
        title))
    plt.legend(loc='best')
    # plt.show()
    plt.draw()  # non blocking in a script
    plt.pause(0.01)
    plt.savefig("Figures/BootstrapModulationSph_"
                "{0}_m{1}_i{2}.pdf".format(title, n_expectation, n_init))
    plt.savefig("Figures/BootstrapModulationSph_"
                "{0}_m{1}_i{2}.png".format(title, n_expectation, n_init))
    return plt


def main():
    """Modulation of empirical Fréchet mean on spherical/projective Fisher data.

    The variance of the Fréchet mean FM_n of a sample of n IID random variables
    of variance Var is decreasing more slowly in a sphere than in a Euclidean
    space. This example computes the  modulation factor
         alpha = Var( FM_n) / ( n * Var)
    for bootstrap samples from a few real-world empirical distributions on the
    sphere S2 and the projective space P2 taken from the book of Fisher, Lewis
    and Embleton 1987.
    """

    # n_expectation = 50  # for test
    n_expectation = 50000  # for production

    # Read the data and transform them into a data matrix of unit vectors
    book = open_workbook("FisherDatasets.xlsx")

    # FisherB2: Almost Euclidean with low number of points
    # Highly concentrated, very low curvature visible:
    # CV should be almost Euclidean
    sheet = book.sheet_by_name("B2")
    FisherB2 = []
    for row in range(1, sheet.nrows):
        # Columns: index dec incl
        index = sheet.cell_value(row, 0)
        decl = float(sheet.cell_value(row, 1))
        incl = float(sheet.cell_value(row, 2))
        print("index {0}: decl={1} incl={2}".format(index, decl, incl))
        FisherB2.append(polar_2_unit_axis(90.0 + incl, 360. - decl))

    mean_B2 = empirical_frechet_mean_random_init_s2(FisherB2, 1)  # 1000)
    stat_dataset_s2(FisherB2, 'Fisher B2', mean_B2)
    plot_dataset_s2(FisherB2, 'Fisher B2', mean_B2)
    # Fisher B2: Var 0.02040048949198819 rad (Stddev 8.18357235244161 deg)
    # Fisher B2: Extent 14.220727879958375 deg / 11.523491168361373 deg
    plot_empirical_frechet_modulation_bootstrap_s2(FisherB2, 'Fisher B2',
                                                   mean_B2, n_init=1,
                                                   n_expectation=n_expectation)
    # planned 1.007 measured 1.005, but uncertainty is relatively high:
    # should be redone with 100 000 bootstrap samples?


    # Fisher B9: Highly concentrated,
    # CV schould be almost Euclidean
    sheet = book.sheet_by_name("B9")
    FisherB9 = []
    for row in range(1, sheet.nrows):
        # Columns: index dec incl
        index = sheet.cell_value(row, 0)
        decl = float(sheet.cell_value(row, 1))
        incl = float(sheet.cell_value(row, 2))
        print("index {0}: decl={1} incl={2}".format(index, decl, incl))
        FisherB9.append(polar_2_unit_axis(90.0 + incl, 360. - decl))
    mean_B9 = empirical_frechet_mean_random_init_s2(FisherB9, 100)  # 1000)
    stat_dataset_s2(FisherB9, 'Fisher B9', mean_B9)
    plot_dataset_s2(FisherB9, 'Fisher B9', mean_B9)
    # Fisher B9: Var 0.05753728121234058 rad (Stddev 13.743498540265614 deg)
    # Fisher B9: Extent 25.663975367020726 deg / 20.346326339517464 deg
    plot_empirical_frechet_modulation_bootstrap_s2(FisherB9, 'Fisher B9',
                                                   mean_B9, n_init=1,
                                                   n_expectation=n_expectation)
    #  planned: 1.016/1.020 measured 1.025

    # Fisher B7: Highly concentrated, should be Euclidean
    sheet = book.sheet_by_name("B7")
    FisherB7 = []
    for row in range(1, sheet.nrows):
        # Columns: index dec incl
        index = sheet.cell_value(row, 0)
        decl = float(sheet.cell_value(row, 1))
        incl = float(sheet.cell_value(row, 2))
        print("index {0}: decl={1} incl={2}".format(index, decl, incl))
        FisherB7.append(polar_2_unit_axis(90.0 + incl, 360. - decl))
    mean_B7 = empirical_frechet_mean_random_init_s2(FisherB7, 100)  # 1000)
    stat_dataset_s2(FisherB7, 'Fisher B7', mean_B7)
    plot_dataset_s2(FisherB7, 'Fisher B7', mean_B7)
    # Fisher B7: Var 0.1206856924106185 rad (Stddev 19.904465765278466 deg)
    # Fisher B7: Extent 44.25142281508782 deg / 17.973096463950736 deg
    plot_empirical_frechet_modulation_bootstrap_s2(FisherB7, 'Fisher B7',
                                                   mean_B7, n_init=1,
                                                   n_expectation=n_expectation)
    # planned 1.04 measured 1.04
    # but incertainty is relatively high:
    # should it be redone with more bootstrapsamples?



    ###########################################################################
    # Non isotropic distributions - Predictions will be over-estimated
    # close to Euclidean 1D: prediction of modulation where there is actually
    # not in practice

    # FisherB4: Almsot linear along an arc of radius +/-45 deg on a small circle
    # Should be close to Euclidean because the strip is too narrow to feel
    # the curvature curvature
    sheet = book.sheet_by_name("B4")
    FisherB4 = []
    for row in range(1, sheet.nrows):
        # Columns: index plunge pl_azimut (deg)
        index = sheet.cell_value(row, 0)
        plunge = float(sheet.cell_value(row, 1))
        azimuth = float(sheet.cell_value(row, 2))
        print("index {0}: plunge={1} plunge azimuth={2}".format(index, plunge,
                                                                azimuth))
        FisherB4.append(polar_2_unit_axis(90.0 + plunge, azimuth))
    mean_B4 = empirical_frechet_mean_random_init_s2(FisherB4, 1000)
    stat_dataset_s2(FisherB4, 'Fisher B4', mean_B4)
    plot_dataset_s2(FisherB4, 'Fisher B4', mean_B4)
    # Fisher B4: Var 0.40478008151475997 rad (Stddev 36.45290965005298 deg)
    # Fisher B4: Extent 53.62326868546807 deg / 11.221098634988639 deg
    plot_empirical_frechet_modulation_bootstrap_s2(FisherB4, 'Fisher B4',
                                                   mean_B4, n_init=1,
                                                   n_expectation=n_expectation)
    # planned 1.13/1.16, measured 1.0 -> no effect

    ###########################################################################
    # With a large dispersion beyond the KKC conditions

    # FisherB15: Projective, rather Gaussian on the northern hemisphere
    sheet = book.sheet_by_name("B15")
    FisherB15 = []
    for row in range(1, sheet.nrows):
        # Columns: index latitude longitude
        index = sheet.cell(row, 0).value
        lat = sheet.cell(row, 1).value
        long = sheet.cell(row, 2).value
        print("index {0}: lat={1} long={2}".format(index, lat, long))
        FisherB15.append(geographical_2_unit_axis(lat, long))
    mean_B15 = empirical_frechet_mean_random_init_s2(FisherB15, 1000)
    stat_dataset_s2(FisherB15, 'Fisher B15', mean_B15)
    plot_dataset_s2(FisherB15, 'Fisher B15', mean_B15)
    # Fisher B15: Var 0.3222337762399972 rad (Stddev 32.524315316835285 deg)
    # Fisher B15: Extent 76.22236184016303 deg / 63.571256094063564 deg
    plot_empirical_frechet_modulation_bootstrap_s2(FisherB15, 'Fisher B15',
                                                   mean_B15, n_init=1,
                                                   n_expectation=n_expectation)

    # FisherB1:  Projective, 14 pts below 45 deg, 36 closer to north pole
    # Not in projective KKC conditions -> Need to add random init
    sheet = book.sheet_by_name("B1")
    FisherB1 = []
    for row in range(1, sheet.nrows):
        # Columns: index lat long
        index = sheet.cell_value(row, 0)
        lat = float(sheet.cell_value(row, 1))
        long = float(sheet.cell_value(row, 2))
        print("index {0}: lat={1} long={2}".format(index, lat, long))
        FisherB1.append(geographical_2_unit_axis(lat, long))
    mean_B1 = empirical_frechet_mean_random_init_s2(FisherB1, 1000)
    stat_dataset_s2(FisherB1, 'Fisher B1', mean_B1)
    plot_dataset_s2(FisherB1, 'Fisher B1', mean_B1)
    # Fisher B1 Projective: Var 0.52141924375251 rad (Stddev 41.3729187320 deg)
    # Fisher B1 Projective: Extent 98.561481928706 deg / 67.82394894422913 deg
    plot_empirical_frechet_modulation_bootstrap_s2(FisherB1, 'Fisher B1',
                                                   mean_B1, n_init=5,
                                                   n_expectation=int(
                                                       n_expectation/5))


    # to avoid exiting
    plt.figure()
    plt.show()


if __name__ == "__main__":
    main()
