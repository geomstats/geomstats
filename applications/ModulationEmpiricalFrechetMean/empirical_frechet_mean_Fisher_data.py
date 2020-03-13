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
from xlrd import open_workbook
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.learning.frechet_mean import _adaptive_gradient_descent


def mean_sq_dist_s2(location, dataset):
    """compute the mean-square deviation from the location to the points of
    the dataset"""
    sphere = Hypersphere(2)
    num_sample = len(dataset)
    assert num_sample > 0, "Dataset needs to have at least one data"
    MSD = 0.0
    for item in dataset:
        sq_dist = sphere.metric.squared_dist(location, item)
        MSD = MSD + sq_dist
    return MSD / num_sample


def var_hbar_s2(location, dataset):
    """compute the mean-square deviation from the location to the points of
    the dataset and the mean Hessian of square distance at the location"""
    sphere = Hypersphere(2)
    num_sample = len(dataset)
    assert num_sample > 0, "Dataset needs to have at least one data"
    var = 0.0
    hbar = 0.0
    for item in dataset:
        sq_dist = sphere.metric.squared_dist(location, item)
        var = var + sq_dist
        # hbar = E(h(dist ^ 2)) with h(t) = sqrt(t) cot( sqrt(t) )  for kappa=1
        if sq_dist > 1e-4:
            d = gs.sqrt(sq_dist)
            h = d / gs.tan(d)
        else:
            h = 1.0 - sq_dist / 3.0 - sq_dist ** 2 / 45.0 - 2 / 945 * \
                sq_dist ** 3 - sq_dist ** 4 / 4725
        hbar = hbar + h
    return var / num_sample, h / num_sample


def cov_hessian_covmean_sphere(mean, dataset):
    """compute dataset covariance, Hessian and covariance of the Fréchet mean"

    Returns
    -------
    Cov: $Cov = 1/n \sum_i=1^n log_mean(x_i) log_mean(x_i)^t$
    Hess: $H = 1/n \sum_i=1^n \partial^2 \dist^2(mean, x_i) / \partial mean^2$
    Hinv: inverse Hessian restricted to the tangent space
    Cov_mean_CLT: covariance predicted on mean by CLT
    Cov_mean_HC: covariance predicted on mean by high concentration expansion
    """
    num_sample = len(dataset)
    assert num_sample > 0, "dataset needs to have at least one data"
    dim = len(dataset[0])
    sphere = Hypersphere(dim)
    Cov = np.zeros((dim, dim), 'float')
    Hess = np.zeros((dim, dim), 'float')
    identity = np.identity(dim, 'float')
    for item in dataset:
        xy = sphere.metric.log(item, mean)
        theta = sphere.metric.norm(xy, mean)
        Cov += np.outer(xy, xy)
        # Numerical issues
        if theta < 1e-3:
            a = 1. / 3. + theta ** 2 / 45. + (2. / 945.) * theta ** 4 \
                + theta ** 6 / 4725 + ( 2. / 93555.) * theta ** 8
            b = 1. - theta ** 2 / 3. - theta ** 4 / 45. \
                - (2. / 945.) * theta ** 6 - theta ** 8 / 4725
        else:
            a = 1. / theta ** 2 - 1.0 / np.tan(theta) / theta
            b = theta / np.tan(theta)
        Hess += a * np.outer(xy, xy) + b * (identity - np.outer(mean, mean))
    Hess = 2 * Hess / num_sample
    Cov = Cov / num_sample
    # ATTN inverse should be restricted to the tangent space because mean is
    # an eigenvector of Hessian with eigenvalue 0
    # Thus, we add a component along mean.mean^t to make eigenvalue 1 before
    # inverting
    Hinv = np.linalg.inv(Hess + np.outer(mean, mean))
    # and we remove the component along mean.mean^t afterward
    Hinv -= np.dot(mean, np.matmul(Hinv, mean)) * np.outer(mean, mean)
    # Compute covariance of the mean predicted
    Cov_mean_CLT = 4 * np.matmul(Hinv, np.matmul(Cov, Hinv)) / num_sample
    Cov_mean_HC = np.matmul(Cov, 2.0 * (identity - (1.0 - 1.0 /
                                                        num_sample) / 3.0
                                      * (Cov - np.trace(Cov) * identity))) / \
                num_sample
    return Cov, Hess, Hinv,  Cov_mean_CLT, Cov_mean_HC


def anisotropic_modulation_factor_sphere(mean, dataset):
    """compute anisotropic modulation factor for Fréchet mean on sphere"

    Returns
    -------
    alpha_CLT: modulation predicted by CLT
    alpha_HC: modulation predicted by high concentration expansion
    """
    (Cov, Hess, Hinv,  Cov_mean_CLT, Cov_mean_HC) = \
        cov_hessian_covmean_sphere(mean, dataset)
    sig2 = np.trace(Cov)
    sig2_HC = np.trace(Cov_mean_HC)
    sig2_CLT = np.trace(Cov_mean_CLT)
    print("Var = {0}, VarMeanSM = {1}, VarMeanCLT = {2}".format(sig2, sig2_HC,
                                                                sig2_CLT))
    alpha_HC = sig2_HC / sig2 * len(dataset)
    alpha_CLT = sig2_CLT / sig2 * len(dataset)
    print(
        "Anisotropic modulation factor: small var {0} / asymptotic {1}".format(
            alpha_HC, alpha_CLT))
    return alpha_HC, alpha_CLT


def max_extent_s2(location, dataset):
    """compute maximal extension of the dataset in two orthogonal directions"""
    sphere = Hypersphere(2)
    num_sample = len(dataset)
    assert num_sample > 0, "Dataset needs to have at least one data"
    max_1 = 0.0
    log_1 = sphere.metric.log(location, location)  # point, base_point
    for item in dataset:
        xy = sphere.metric.log(item, location)
        theta = sphere.metric.norm(xy, location)
        if theta > max_1:
            max_1 = theta
            log_1 = xy
    log_1 = log_1 / sphere.metric.norm(log_1, location)
    max_2 = 0.0
    for item in dataset:
        xy = sphere.metric.log(item, location)
        # project to get the orthogonal part to Log_1
        xy = xy - sphere.metric.inner_product(xy, log_1, location) * log_1
        theta = sphere.metric.norm(xy, location)
        if theta > max_2:
            max_2 = theta
    return max_1, max_2


def stat_dataset_s2(dataset, title, frechet_mean):
    """Print concentration statistics on a dataset on the sphere S2"

    Parameters
    ----------
    dataset: spherical samples to be drawn
    title: title of dataset
    frechet_mean: list of Fréchet mean points
    """
    assert len(frechet_mean) > 0, "Fréchet mean or reference point needs to " \
                                  "be provided"

    # print some statistics on the concentration of the dataset
    (var, hbar) = var_hbar_s2(frechet_mean[0], dataset)
    print("{2}: Var {0} rad (Stddev {1} deg)".format(var, np.sqrt(
        var) * 180 / gs.pi, title))
    (max1, max2) = max_extent_s2(frechet_mean[0], dataset)
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
    plt.show()
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


def empirical_frechet_var_bootstrap(n_samples, theta, dim,
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
    theta: radius of the bubble distribution
    dim: dimension of the sphere (embedded in R^{dim+1})
    n_expectation: number of computations for approximating the expectation

    Returns
    -------
    tuple (variance, std-dev on the computed variance)
    """
def ComputeBootstrapFrechetCV(Dataset, title, FM, M=1000, N_init=1,
                              NumSample=[2, 100]):
    # Dataset = original dataset to study
    # M = number of points for stochastic expectation
    # should start with many initial points and keep the best
    # FM = EmpiricalFrechetMeanRandomInit(Dataset, 1000)
    (var, hbar) = var_hbar(FM, Dataset)
    (Cov, CovMeanSM, CovMeanCLT) = Cov_CovMean(FM, Dataset)
    Id = np.identity(len(Dataset[0]), 'float')

    MeanAlpha = []
    StdAlpha = []
    PredictedFactor = []
    AsymptoticFactor = []
    AnisoPredictedFactor = []
    AnisoAsymptoticFactor = []
    dim = len(Dataset[0]) - 1
    for n_samples in NumSample:
        alphalist = []
        # h = []
        for i in range(M):
            # bootstrap: Sample n_samples points from the empirical distribution DataList
            # BootstrapDataset = random.sample(Dataset, n_samples)
            BootstrapDataset = []
            for j in range(n_samples):
                BootstrapDataset.append(random.choice(Dataset))
            # FM_i = EmpiricalFrechetMean( BootstrapDataset, [FM] )
            FM_i = EmpiricalFrechetMeanRandomInit(BootstrapDataset, N_init,
                                                  [FM])
            di = Dist(FM_i, FM)
            alphalist.append(di ** 2 / var * n_samples)
            # h.append( di / np.tan(di))  # hbar = E(h(dist^2)) for a spere or kappa=1
            # print( di**2 /var * n_samples )
            # print("h = {0}".format(di / np.tan(di)))
            # print( "di = {0}   h = {1}".format(di, di/ np.tan(di)))
        # get mean_alpha and stddev of mean_alpha
        mean = np.mean(alphalist)
        stddev = np.std(alphalist) / np.sqrt(M - 1.0)
        MeanAlpha.append(mean)
        StdAlpha.append(stddev)
        Predicted = 1.0 + kappa * 2.0 / 3.0 * var * (1.0 - 1.0 / dim) * (
                    1.0 - 1.0 / (n_samples))
        PredictedFactor.append(Predicted)
        # hbar = np.mean(h)
        # print("hbar = {0}".format(hbar))
        Asymptotic = (1.0 / dim + (1.0 - 1.0 / dim) * hbar) ** (-2)
        AsymptoticFactor.append(Asymptotic)
        print(
            "{0} samples : Modulation ratio = {1} pm {2} / Small var prediction {3} / Asymptotic {4}".format(
                n_samples, mean, stddev, Predicted, Asymptotic))
        AnisoCov = np.matmul(Cov, (
                    Id - 2.0 * kappa * (1.0 - 1.0 / n_samples) / 3.0 * (
                        Cov - np.trace(Cov) * Id))) / n_samples
        alpha_smallVar = np.trace(AnisoCov) / var * n_samples
        AnisoPredictedFactor.append(alpha_smallVar)
        alpha_clt = np.trace(CovMeanCLT) / var * len(Dataset)
        AnisoAsymptoticFactor.append(alpha_clt)
        print(
            "{0} samples : Anisotropic pridiction  Small var  {1} / Asymptotic {2}".format(
                n_samples, alpha_smallVar, alpha_clt))

    return (MeanAlpha, StdAlpha, PredictedFactor, AnisoPredictedFactor,
            AsymptoticFactor, AnisoAsymptoticFactor)


#
# def empirical_frechet_var_bubble(n_samples, theta, dim,
#                                  n_expectation=1000):
#     """Variance of the empirical Fréchet mean for a bubble distribution.
#
#     Draw n_sampless from a bubble distribution, computes its empirical
#     Fréchet mean and the square distance to the asymptotic mean. This
#     is repeated n_expectation times to compute an approximation of its
#     expectation (i.e. its variance) by sampling.
#
#     The bubble distribution is an isotropic distributions on a Riemannian
#     hyper sub-sphere of radius 0 < theta < Pi around the north pole of the
#     sphere of dimension dim.
#
#     Parameters
#     ----------
#     n_samples: number of samples to draw
#     theta: radius of the bubble distribution
#     dim: dimension of the sphere (embedded in R^{dim+1})
#     n_expectation: number of computations for approximating the expectation
#
#     Returns
#     -------
#     tuple (variance, std-dev on the computed variance)
#     """
#     assert dim > 1, "Dim > 1 needed to draw a uniform sample on sub-sphere"
#     var = []
#     sphere = Hypersphere(dimension=dim)
#     bubble = Hypersphere(dimension=dim - 1)
#
#     north_pole = gs.zeros(dim + 1)
#     north_pole[dim] = 1.0
#     for k in range(n_expectation):
#         # Sample n points from the uniform distribution on a sub-sphere
#         # of radius theta (i.e cos(theta) in ambient space)
#         # TODO(nina): Add this code as a method of hypersphere
#         data = gs.zeros((n_samples, dim + 1), dtype=gs.float64)
#         directions = bubble.random_uniform(n_samples)
#
#         for i in range(n_samples):
#             for j in range(dim):
#                 data[i, j] = gs.sin(theta) * directions[i, j]
#             data[i, dim] = gs.cos(theta)
#         # TODO(nina): Use FrechetMean here
#         current_mean = _adaptive_gradient_descent(
#             data, metric=sphere.metric,
#             n_max_iterations=64, init_points=[north_pole])
#         var.append(sphere.metric.squared_dist(north_pole, current_mean))
#     return np.mean(var), 2 * np.std(var) / gs.sqrt(n_expectation)
#
#
# def modulation_factor(n_samples, theta, dim, n_expectation=1000):
#     """Modulation factor on the convergence of the empirical Fréchet mean.
#
#     The modulation factor is the ratio of the variance of the empirical
#     Fréchet mean on the manifold to the variance in a Euclidean space,
#     for n_sampless drawn from an isotropic distributions on a Riemannian
#     hyper sub-sphere of radius 0 < theta < Pi around the north pole of the
#     sphere of dimension dim.
#
#     Parameters
#     ----------
#     n_samples: number of samples to draw
#     theta: radius of the bubble distribution
#     dim: dimension of the sphere (embedded in R^{dim+1})
#     n_expectation: number of computations for approximating the expectation
#
#     Returns
#     -------
#     tuple (modulation factor, std-dev on the modulation factor)
#     """
#     (var, std_var) = empirical_frechet_var_bubble(
#         n_samples, theta, dim, n_expectation=n_expectation)
#     return var * n_samples / theta ** 2, std_var * n_samples / theta ** 2
#
#
# def asymptotic_modulation(dim, theta):
#     """Compute the asymptotic modulation factor.
#
#     Parameters
#     ----------
#     dim: dimension of the sphere (embedded in R^{dim+1})
#     theta: radius of the bubble distribution
#
#     Returns
#     -------
#     tuple (modulation factor, std-dev on the modulation factor)
#     """
#     gamma = 1.0 / dim + (1.0 - 1.0 / dim) * theta / gs.tan(theta)
#     return (1.0 / gamma) ** 2
#
#
# def plot_modulation_factor(n_samples, dim, n_expectation=1000, n_theta=20):
#     """Plot the modulation factor curve w.r.t. the dispersion.
#
#     Plot the curve of modulation factor on the convergence of the
#     empirical Fréchet mean as a function of the radius of the bubble
#     distribution and for n_samples points on the sphere S_dim
#     embedded in R^{dim+1}.
#
#     Parameters
#     ----------
#     n_samples: number of samples to draw
#     dim: dimension of the sphere (embedded in R^{dim+1})
#     n_expectation: number of computations for approximating the expectation
#     n_theta: number of sampled radii for the bubble distribution
#
#     Returns
#     -------
#     matplolib figure
#     """
#     theta = gs.linspace(0.000001, gs.pi / 2.0 - 0.000001, n_theta)
#     measured_modulation_factor = []
#     error = []
#     small_var_modulation_factor = []
#     asymptotic_modulation_factor = []
#     for theta_i in theta:
#         (var, std_var) = modulation_factor(
#             n_samples, theta_i, dim, n_expectation=n_expectation)
#         measured_modulation_factor.append(var)
#         error.append(std_var)
#         print('{} {} {} {}\n'.format(n_samples, theta_i, var, std_var))
#         small_var_modulation_factor.append(
#             1.0 + 2.0 / 3.0 * theta_i ** 2
#             * (1.0 - 1.0 / dim) * (1.0 - 1.0 / n_samples))
#         asymptotic_modulation_factor.append(
#             asymptotic_modulation(dim, theta_i))
#     plt.figure()
#     plt.errorbar(theta, measured_modulation_factor,
#                  yerr=error, color='r', label='Measured')
#     plt.plot(theta, small_var_modulation_factor,
#              'g', label='Small variance prediction')
#     plt.plot(theta, asymptotic_modulation_factor,
#              'grey', label='Asymptotic prediction')
#     plt.xlabel(r'Standard deviation $\theta$')
#     plt.ylabel(r'Modulation factor $\alpha$')
#     plt.title("Convergence rate modulation factor, "
#               "sphere dim={1}, n={0}".format(n_samples, dim))
#     plt.legend(loc='best')
#     plt.draw()
#     plt.pause(0.01)
#     plt.savefig("Figures/SphVarModulation_N{0}_d{1}_m{2}.png".format(
#         n_samples, dim, n_expectation))
#     plt.savefig("Figures/SphVarModulation_N{0}_d{1}_m{2}.pdf".format(
#         n_samples, dim, n_expectation))
#     return plt


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

    n_expectation = 50  # for test # use 50 000 for production

    ## Read the data and transform them into a data matrix of unit vectors
    book = open_workbook("FisherDatasets.xlsx")

    # FisherB2: Almost Euclidean with low number of points
    # Highly concentrated, very low curvature visible:
    # CV should be almost Euclidean
    sheet = book.sheet_by_name("B2")
    FisherB2 = []
    for row in range(1, sheet.nrows):  ## start at row 1 to avoid titles...
        # Columns: index dec incl
        index = sheet.cell_value(row, 0)
        decl = float(sheet.cell_value(row, 1))
        incl = float(sheet.cell_value(row, 2))
        print("index {0}: decl={1} incl={2}".format(index, decl, incl))
        FisherB2.append(polar_2_unit_axis(90.0 + incl, 360. - decl))

    mean_B2 = empirical_frechet_mean_random_init_s2(FisherB2, 1000)
    stat_dataset_s2(FisherB2, 'Fisher B2', mean_B2)
    plot_dataset_s2(FisherB2, 'Fisher B2', mean_B2)
    # Fisher B2: Var 0.02040048949198819 rad (Stddev 8.18357235244161 deg)
    # Fisher B2: Extent 14.220727879958375 deg / 11.523491168361373 deg
    # ComputeBootstrapFrechetCV(FisherB2, "Fisher B2", mean_B2, 5000)
    # PlotBootstrapFrechetCV(FisherB2, "Fisher B2", mean_B2, n_expectation)
    # PlotBootstrapFrechetCV(FisherB2, "Fisher B2", FMB2, 50000) # for production
    # planned 1.007 measured 1.005, but uncertainty is relatively high:
    # should be redone with 100 000 bootstrap samples?


if __name__ == "__main__":
    main()
