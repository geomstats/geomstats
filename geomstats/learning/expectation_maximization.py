"""Expectation maximization algorithm."""

import logging

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.learning.frechet_mean import variance
from geomstats.learning.kmeans import RiemannianKMeans

EM_CONV_RATE = 1e-4
MINIMUM_EPOCHS = 10
DEFAULT_MAX_ITER = 100
DEFAULT_LR = 5e-2
DEFAULT_CONV_FACTOR = 5e-3
DEFAULT_TOL = 1e-2
ZETA_LOWER_BOUND = 5e-1
ZETA_UPPER_BOUND = 2.
ZETA_STEP = 0.01
PDF_TOL = 1e-6
SUM_CHECK_PDF = 1e-4
MEAN_MAX_ITER = 150
MIN_VAR_INIT = 1e-3


def gmm_pdf(
        data, means, variances, norm_func,
        metric, variances_range, norm_func_var):
    """Return the separate probability density function of GMM.

    The probability density function is computed for
    each component of the GMM separately (i.e., mixture coefficients
    are not taken into account).

    Parameters
    ----------
    data : array-like, shape=[n_samples, dim]
        Points at which the GMM probability density is computed.
    means : array-like, shape=[n_gaussians, dim]
        Means of each component of the GMM.
    variances : array-like, shape=[n_gaussians,]
        Variances of each component of the GMM.
    norm_func : function
        Normalisation factor function.
    metric : RiemannianMetric
        Metric to use on the manifold.

    Returns
    -------
    pdf : array-like, shape=[n_samples, n_gaussians,]
        Probability density function computed at each data
        sample and for each component of the GMM.
    """
    data_length, _, _ = data.shape + (means.shape[0],)

    variances_expanded = gs.expand_dims(variances, 0)
    variances_expanded = gs.repeat(variances_expanded, data_length, 0)

    variances_flatten = variances_expanded.flatten()

    distances = -(metric.dist_broadcast(data, means) ** 2)
    distances = gs.reshape(distances, (data.shape[0] * variances.shape[0]))

    num = gs.exp(
        distances / (2 * variances_flatten ** 2))

    den = norm_func(variances, variances_range, norm_func_var)

    den = gs.expand_dims(den, 0)
    den = gs.repeat(den, data_length, axis=0).flatten()

    pdf = num / den
    pdf = gs.reshape(
        pdf, (data.shape[0], means.shape[0]))

    return pdf


def weighted_gmm_pdf(
        mixture_coefficients, mesh_data, means, variances, metric):
    """Return the probability density function of a GMM.

    Parameters
    ----------
    mixture_coefficients : array-like, shape=[n_gaussians,]
        Coefficients of the Gaussian mixture model.
    mesh_data : array-like, shape=[n_precision, dim]
        Points at which the GMM probability density is computed.
    means : array-like, shape=[n_gaussians, dim]
        Means of each component of the GMM.
    variances : array-like, shape=[n_gaussians,]
        Variances of each component of the GMM.
    metric : RiemannianMetric
        Metric to use on the manifold.

    Returns
    -------
    weighted_pdf : array-like, shape=[n_precision, n_gaussians,]
        Probability density function computed for each point of
        the mesh data, for each component of the GMM.
    """
    distance_to_mean = metric.dist_broadcast(mesh_data, means)

    variances_units = gs.expand_dims(variances, 0)
    variances_units = gs.repeat(
        variances_units, distance_to_mean.shape[0], axis=0)

    distribution_normal = gs.exp(
        -(distance_to_mean ** 2) / (2 * variances_units ** 2))

    zeta_sigma = (2 * gs.pi) ** (2 / 3) * variances
    zeta_sigma = zeta_sigma * gs.exp(
        (variances ** 2 / 2) * gs.erf(variances / gs.sqrt(2)))

    result_num = gs.expand_dims(mixture_coefficients, 0)
    result_num = gs.repeat(result_num, len(distribution_normal), axis=0)
    result_num = result_num * distribution_normal

    result_denum = gs.expand_dims(zeta_sigma, 0)
    result_denum = gs.repeat(result_denum, len(distribution_normal), axis=0)

    weighted_pdf = result_num / result_denum

    return weighted_pdf


def find_normalization_factor(
        variances, variances_range, normalization_factor_var):
    """Find the normalization factor given some variances.

    Parameters
    ----------
    variances : array-like, shape=[n_gaussians,]
        Array of standard deviations for each component
        of some GMM.
    variances_range : array-like, shape=[n_variances,]
        Array of standard deviations.
    normalization_factor_var : array-like, shape=[n_variances,]
        Array of computed normalization factor.

    Returns
    -------
    norm_factor : array-like, shape=[n_gaussians,]
        Array of normalization factors for the given
        variances.
    """
    n_gaussians, precision = variances.shape[0], variances_range.shape[0]

    ref = gs.expand_dims(variances_range, 0)
    ref = gs.repeat(ref, n_gaussians, axis=0)
    val = gs.expand_dims(variances, 1)
    val = gs.repeat(val, precision, axis=1)

    difference = gs.abs(ref - val)

    index = gs.argmin(difference, axis=-1)
    norm_factor = normalization_factor_var[index]

    return norm_factor


def find_variance_from_index(
        weighted_distances, variances_range, phi_inv_var):
    r"""Return the variance given weighted distances.

    Parameters
    ----------
    weighted_distances : array-like, shape=[n_gaussians,]
        Mean of the weighted distances between training data
        and current barycentres. The weights of each data sample
        corresponds to the probability of belonging to a component
        of the Gaussian mixture model.
    variances_range : array-like, shape=[n_variances,]
        Array of standard deviations.
    phi_inv_var : array-like, shape=[n_variances,]
        Array of the computed inverse of a function phi
        whose expression is closed-form
        :math:`\sigma\mapsto \sigma^3 \times \frac{d  }
        {\mathstrut d\sigma}\log \zeta_m(\sigma)'
        where :math:'\sigma' denotes the variance
        and :math:'\zeta' the normalization coefficient
        and :math:'m' the dimension.

    Returns
    -------
    var : array-like, shape=[n_gaussians,]
        Estimated variances for each component of the GMM.
    """
    n_gaussians, precision = \
        weighted_distances.shape[0], variances_range.shape[0]

    ref = gs.expand_dims(phi_inv_var, 0)
    ref = gs.repeat(ref, n_gaussians, axis=0)

    val = gs.expand_dims(weighted_distances, 1)
    val = gs.repeat(val, precision, axis=1)

    abs_difference = gs.abs(ref - val)

    index = gs.argmin(abs_difference, -1)

    var = variances_range[index]

    return var


class RiemannianEM(TransformerMixin, ClusterMixin, BaseEstimator):
    r"""Expectation-maximization algorithm.

    A class for performing Expectation-Maximization to fit a Gaussian Mixture
    Model (GMM) to data on a manifold. This method is only implemented for
    the hypersphere and the Poincare ball.

    Parameters
    ----------
    n_gaussians : int
        Number of Gaussian components in the mix.
    metric : object of class RiemannianMetric
        The geomstats Riemmanian metric associated with
        the used manifold.
    initialisation_method : basestring
        Optional, default: 'random'.
        Choice between initialization method for variances, means and weights.
           'random' : will select random uniformly train points as
                     initial centroids.
            'kmeans' : will apply Riemannian kmeans to deduce
            variances and means that the EM will use initially.
    tol : float
        Optional, default: 1e-2.
        Convergence tolerance. If the difference of mean distance
        between two steps is lower than tol.
    lr_mean : float
        Learning rate in the gradient descent computation of the Frechet means.
        Optional, default: 1.
    max_iter : int
        Maximum number of iterations for the gradient descent.
        Optional, default: 100.
    max_iter_mean : int
        Maximum number of iterations for the gradient descent of each Frechet
        mean.
        Optional, default: 100.
    tol_mean : float, optional
        Tolerance for stopping the gradient descent of the computation of the
        Frechet mean.
        Optional, default: 1e-4.

    Attributes
    ----------
    point_type : basestring
        Whether to use vector or matrix representation.
    _dimension : int
        Manifold dimension.
    mixture_coefficients : array-like, shape=[n_gaussians,]
        Weights for each GMM component.
    variances : array-like, shape=[n_gaussians,]
        Variances for each GMM component.
    means : array-like, shape=[n_gaussian, _dimension]
        Barycentre of each component of the GMM.
    normalization_factor_var : array-like, shape=[n_variances,]
        Array of computed normalization factor.
    variances_range : array-like, shape=[n_variances,]
        Array of standard deviations.
    phi_inv_var : array-like, shape=[n_variances,]
        Array of the computed inverse of a function phi
        whose expression is closed-form
        :math:`\sigma\mapsto \sigma^3 \times \frac{d  }
        {\mathstrut d\sigma}\log \zeta_m(\sigma)'
        where :math:'\sigma' denotes the variance
        and :math:'\zeta' the normalization coefficient
        and :math:'m' the dimension.

    Example
    -------
    Available example on the Poincaré Ball manifold
    :mod:`examples.plot_expectation_maximization_ball`
    """

    def __init__(self,
                 metric,
                 n_gaussians=8,
                 initialisation_method='random',
                 tol=DEFAULT_TOL,
                 lr_mean=1.,
                 max_iter=100,
                 max_iter_mean=100,
                 tol_mean=1e-4):

        self.n_gaussians = n_gaussians
        self.metric = metric
        self.initialisation_method = initialisation_method
        self.tol = tol
        self.mean_method = 'batch'
        self.point_type = metric.default_point_type
        self._dimension = None
        self.mixture_coefficients = None
        self.variances = None
        self.means = None
        self.normalization_factor = None
        self.variances_range = None
        self.normalization_factor_var = None
        self.phi_inv_var = None
        self.lr_mean = lr_mean
        self.max_iter = max_iter
        self.max_iter_mean = max_iter_mean
        self.tol_mean = tol_mean

    def update_posterior_probabilities(self, posterior_probabilities):
        """Posterior probabilities update function.

        Parameters
        ----------
        posterior_probabilities : array-like, shape=[n_samples, n_gaussians,]
            Probability of a given sample to belong to a component
            of the GMM, computed for all components.
        """
        self.mixture_coefficients = gs.mean(posterior_probabilities, 0)

    def update_means(
            self, data, posterior_probabilities):
        """Update means."""
        n_gaussians = posterior_probabilities.shape[-1]

        mean = FrechetMean(
            metric=self.metric,
            method=self.mean_method,
            lr=self.lr_mean,
            epsilon=self.tol_mean,
            max_iter=self.max_iter_mean,
            point_type=self.point_type)

        data_expand = gs.expand_dims(data, 1)
        data_expand = gs.repeat(data_expand, n_gaussians, axis=1)

        mean.fit(data_expand, weights=posterior_probabilities)
        self.means = gs.squeeze(mean.estimate_)

    def update_variances(self, data, posterior_probabilities):
        """Update variances function.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features,]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        posterior_probabilities : array-like, shape=[n_samples, n_gaussians,]
            Probability of a given sample to belong to a component
            of the GMM, computed for all components.
        """
        dist_means_data = (self.metric.dist_broadcast(
            data, self.means) ** 2)

        weighted_dist_means_data = (dist_means_data *
                                    posterior_probabilities).sum(0) / \
            posterior_probabilities.sum(0)

        self.variances = find_variance_from_index(
            weighted_dist_means_data, self.variances_range, self.phi_inv_var)

    def _expectation(self, data):
        """Update the posterior probabilities.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        """
        probability_distribution_function = gmm_pdf(
            data, self.means, self.variances,
            norm_func=find_normalization_factor,
            metric=self.metric,
            variances_range=self.variances_range,
            norm_func_var=self.normalization_factor_var)

        if gs.isnan(probability_distribution_function.mean()):
            logging.warning('EXPECTATION : Probability distribution function'
                            'contain elements that are not numbers')

        num_normalized_pdf = gs.einsum('j,...j->...j',
                                       self.mixture_coefficients,
                                       probability_distribution_function)
        valid_pdf_condition = gs.amin(gs.sum(num_normalized_pdf, -1))

        if valid_pdf_condition <= PDF_TOL:

            num_normalized_pdf[gs.sum(num_normalized_pdf, -1) <= PDF_TOL] = 1

        sum_pdf = gs.sum(num_normalized_pdf, -1)
        posterior_probabilities =\
            gs.einsum('...i,...->...i', num_normalized_pdf, 1 / sum_pdf)

        if gs.any(gs.mean(posterior_probabilities)) is None:

            logging.warning('EXPECTATION : posterior probabilities '
                            'contain elements that are not numbers.')

        if 1 - SUM_CHECK_PDF >= gs.mean(gs.sum(
                posterior_probabilities, 1)) >= 1 + SUM_CHECK_PDF:

            logging.warning('EXPECTATION : posterior probabilities '
                            'do not sum to 1.')

        if gs.any(gs.sum(posterior_probabilities, 0) < PDF_TOL):
            logging.warning('EXPECTATION : Gaussian got no elements '
                            '(precision error) reinitialize')
            posterior_probabilities[posterior_probabilities == 0] = PDF_TOL

        return posterior_probabilities

    def _maximization(self, data, posterior_probabilities):
        """Update function for the means and variances.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features,]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        posterior_probabilities : array-like, shape=[n_samples, n_gaussians,]
            Probability of a given sample to belong to a component
            of the GMM, computed for all components.
        """
        self.update_posterior_probabilities(posterior_probabilities)

        if(gs.mean(self.mixture_coefficients)
                != gs.mean(self.mixture_coefficients)):
            logging.warning('UPDATE : mixture coefficients '
                            'contain elements that are not numbers')

        self.update_means(data, posterior_probabilities)

        if self.means.mean() != self.means.mean():
            logging.warning('UPDATE : means contain'
                            'not a number elements')

        self.update_variances(data, posterior_probabilities)

        if self.variances.mean() != self.variances.mean():
            logging.warning('UPDATE : variances contain'
                            'not a number elements')

    def fit(self, data):
        """Fit a Gaussian mixture model (GMM) given the data.

        Alternates between Expectation and Maximization steps
        for some number of iterations.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Return the components of the computed
            Gaussian mixture model: means, variances and mixture_coefficients.
        """
        self._dimension = data.shape[-1]
        if self.initialisation_method == 'kmeans':
            kmeans = RiemannianKMeans(
                metric=self.metric, n_clusters=self.n_gaussians, init='random',
                mean_method='batch', lr=self.lr_mean)

            centroids = kmeans.fit(X=data)
            labels = kmeans.predict(X=data)

            self.means = centroids
            self.variances = gs.zeros(self.n_gaussians)

            labeled_data = gs.vstack([labels, gs.transpose(data)])
            labeled_data = gs.transpose(labeled_data)
            for label, centroid in enumerate(centroids):
                label_mask = gs.where(labeled_data[:, 0] == label)
                grouped_by_label = labeled_data[label_mask][:, 1:]
                v = variance(grouped_by_label, centroid, self.metric)
                if grouped_by_label.shape[0] == 1:
                    v += MIN_VAR_INIT
                self.variances[label] = v
        else:
            self.means = (gs.random.rand(
                self.n_gaussians,
                self._dimension) - 0.5) / self._dimension
            self.variances = gs.random.rand(self.n_gaussians) / 10 + 0.8

        self.mixture_coefficients = \
            gs.ones(self.n_gaussians) / self.n_gaussians
        posterior_probabilities = gs.ones((data.shape[0],
                                           self.means.shape[0]))

        self.variances_range,\
            self.normalization_factor_var, \
            self.phi_inv_var =\
            self.normalization_factor_init(
                gs.arange(
                    ZETA_LOWER_BOUND, ZETA_UPPER_BOUND, ZETA_STEP))

        for epoch in range(self.max_iter):
            old_posterior_probabilities = posterior_probabilities

            posterior_probabilities = self._expectation(data)

            condition = gs.mean(gs.abs(old_posterior_probabilities
                                       - posterior_probabilities))

            if condition < EM_CONV_RATE and epoch > MINIMUM_EPOCHS:
                logging.info('EM converged in %s iterations', epoch)
                return self.means, self.variances, self.mixture_coefficients

            self._maximization(data, posterior_probabilities)

        logging.info('WARNING: EM did not converge \n'
                     'Please increase MINIMUM_EPOCHS.')

        return self.means, self.variances, self.mixture_coefficients

    def normalization_factor_init(self, variances):
        r"""Set up function for the normalization factor.

        The normalization factor is used to define Gaussian distributions
        at initialization..

        Parameters
        ----------
        variances : array-like, shape=[n_variances,]
            Array of standard deviations.
        normalization_factor_var : array-like, shape=[n_variances,]
            Array of computed normalization factor.
        phi_inv_var : array-like, shape=[n_variances,]
            Array of the computed inverse of a function phi
            whose expression is closed-form
            :math:`\sigma\mapsto \sigma^3 \times \frac{d  }
            {\mathstrut d\sigma}\log \zeta_m(\sigma)'
            where :math:'\sigma' denotes the variance
            and :math:'\zeta' the normalization coefficient
            and :math:'m' the dimension.

        Returns
        -------
        variances : array-like, shape=[n_variances,]
            Array of standard deviations.
        normalization_factor_var : array-like, shape=[n_variances,]
            Array of computed normalization factor.
        phi_inv_var : array-like, shape=[n_variances,]
            Array of the computed inverse of a function phi
            whose expression is closed-form
            :math:`\sigma\mapsto \sigma^3 \times \frac{d  }
            {\mathstrut d\sigma}\log \zeta_m(\sigma)'
            where :math:'\sigma' denotes the variance
            and :math:'\zeta' the normalization coefficient
            and :math:'m' the dimension.
        """
        normalization_factor_var = \
            self.metric.normalization_factor(variances)

        cond_1 = normalization_factor_var.sum() != \
            normalization_factor_var.sum()
        cond_2 = normalization_factor_var.sum() == float('+inf')
        cond_3 = normalization_factor_var.sum() == float('-inf')

        if cond_1 or cond_2 or cond_3:
            logging.warning(
                'Untracktable normalization factor :')

            limit_nf = (
                (normalization_factor_var / normalization_factor_var)
                * 0).nonzero()[0].item()
            max_nf = len(variances)
            variances = variances[0:limit_nf]
            normalization_factor_var = \
                normalization_factor_var[0:limit_nf]
            if cond_1:
                logging.warning('\t Nan value '
                                'in processing normalization factor')
            if cond_2 or cond_3:
                raise ValueError('\t +-inf value in '
                                 'processing normalization factor')

            logging.warning('\t Max variance is now : %s',
                            str(variances[-1]))
            logging.warning('\t Number of possible variance is now: %s / %s ',
                            str(len(variances)), str(max_nf))

        _, log_grad_zeta = \
            self.metric.norm_factor_gradient(variances)

        phi_inv_var = variances ** 3 * log_grad_zeta

        return \
            variances, normalization_factor_var, phi_inv_var
