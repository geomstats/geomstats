"""Expectation maximization algorithm."""

import logging

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.geometry.poincare_ball \
    import PoincareBall
from geomstats.learning._template import TransformerMixin
from geomstats.learning.frechet_mean import FrechetMean


EM_CONV_RATE = 1e-4
MINIMUM_EPOCHS = 10
DEFAULT_MAX_ITER = 100
DEFAULT_LR = 5e-2
DEFAULT_CONV_FACTOR = 5e-3
DEFAULT_TOL = 1e-2
ZETA_LOWER_BOUND = 5e-2
ZETA_UPPER_BOUND = 2.
ZETA_STEP = 0.001
PDF_TOL = 1e-15
SUM_CHECK_PDF = 1e-4
MEAN_MAX_ITER = 150


class RiemannianEM(TransformerMixin, ClusterMixin, BaseEstimator):
    r"""Expectation-maximization class on Poincaré Ball.

    A class for performing Expectation-Maximization on the
    Poincaré Ball to fit data into a Gaussian Mixture Model (GMM).

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
           'random' : will select random uniformally train point as
                     initial centroids.
            'kmeans' : will apply Riemannian kmeans to deduce
            variances and means that the EM will use initially.
    tol : float
        Optional, default: 1e-2.
        Convergence factor. If the difference of mean distance
        between two step is lower than tol.
    mean_method : basestring
        Optional, default: 'default'.
        Specify the method to compute the mean.
    point_type : basestring
        Optional, default: 'vector'.
        Specify whether to use vector or matrix representation.
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

    Returns
    -------
    self : object
        Returns the instance itself.

    Example
    -------
    Available example on the Poincaré Ball manifold
    :mod:`examples.plot_expectation_maximization_manifolds`
    """

    def __init__(self,
                 metric,
                 n_gaussians=8,
                 initialisation_method='random',
                 tol=DEFAULT_TOL,
                 mean_method='default',
                 point_type='vector'):

        self.n_gaussians = n_gaussians
        self.metric = metric
        self.initialisation_method = initialisation_method
        # TODO : hzaatiti, tgeral68 implement kmeans initialisation
        self.tol = tol
        self.mean_method = mean_method
        self.point_type = point_type
        self._dimension = None
        self.mixture_coefficients = None
        self.variances = None
        self.means = None
        self.normalization_factor = None
        self.variances_range = None
        self.normalization_factor_var = None
        self.phi_inv_var = None

    def update_posterior_probabilities(self, posterior_probabilities):
        """Posterior probabilities update function.

        Parameters
        ----------
        posterior_probabilities : array-like, shape=[n_samples, n_gaussians,]
            Probability of a given sample to belong to a component
            of the GMM, computed for all components.
        """
        self.mixture_coefficients = gs.mean(posterior_probabilities, 0)

    def update_means(self, data, posterior_probabilities,
                     lr_means, tau_means, max_iter=DEFAULT_MAX_ITER):
        """Means update function."""
        n_gaussians = posterior_probabilities.shape[-1]

        mean = FrechetMean(
            metric=self.metric,
            method=self.mean_method,
            lr=lr_means,
            tau=tau_means,
            max_iter=max_iter,
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

        self.variances = \
            self.metric.find_variance_from_index(
                weighted_dist_means_data,
                self.variances_range,
                self.phi_inv_var)

    def _expectation(self, data):
        """Update the posterior probabilities.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        """
        probability_distribution_function = \
            PoincareBall.gmm_pdf(
                data, self.means, self.variances,
                norm_func=self.metric.find_normalization_factor,
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

        return posterior_probabilities

    def _maximization(self,
                      data,
                      posterior_probabilities,
                      lr_means,
                      conv_factor_mean,
                      max_iter=DEFAULT_MAX_ITER):
        """Update function for the means and variances.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features,]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        posterior_probabilities : array-like, shape=[n_samples, n_gaussians,]
            Probability of a given sample to belong to a component
            of the GMM, computed for all components.
        lr_means : float
            Learning rate for computing the means.
        conv_factor_mean : float
            Convergence factor for means.
        max_iter : int
            Optional, default: 100.
            Maximum number of iterations for computing
            the means.
        """
        self.update_posterior_probabilities(posterior_probabilities)

        if(gs.mean(self.mixture_coefficients)
                != gs.mean(self.mixture_coefficients)):
            logging.warning('UPDATE : mixture coefficients '
                            'contain elements that are not numbers')

        self.update_means(data,
                          posterior_probabilities,
                          lr_means=lr_means,
                          tau_means=conv_factor_mean,
                          max_iter=max_iter)

        if self.means.mean() != self.means.mean():
            logging.warning('UPDATE : means contain'
                            'not a number elements')

        self.update_variances(data, posterior_probabilities)

        if self.variances.mean() != self.variances.mean():
            logging.warning('UPDATE : variances contain'
                            'not a number elements')

    def fit(self,
            data,
            max_iter=DEFAULT_MAX_ITER,
            lr_mean=DEFAULT_LR,
            conv_factor_mean=DEFAULT_CONV_FACTOR):
        """Fit a Gaussian mixture model (GMM) given the data.

        Alternates between Expectation and Maximization steps
        for some number of iterations.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        max_iter : int
            Optional, default: 100.
            Maximum number of iterations.
        lr_mean : float
            Optional, default: 5e-2.
            Learning rate for the mean.
        conv_factor_mean : float
            Optional, default: 5e-3.
            Convergence factor for the mean.

        Returns
        -------
        self : object
            Return the components of the computed
            Gaussian mixture model: means, variances and mixture_coefficients.
        """
        self._dimension = data.shape[-1]
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
            self.metric.normalization_factor_init(
                gs.arange(
                    ZETA_LOWER_BOUND, ZETA_UPPER_BOUND, ZETA_STEP))

        for epoch in range(max_iter):
            old_posterior_probabilities = posterior_probabilities

            posterior_probabilities = self._expectation(data)

            condition = gs.mean(gs.abs(old_posterior_probabilities
                                       - posterior_probabilities))

            if(condition < EM_CONV_RATE and epoch > MINIMUM_EPOCHS):
                logging.info('EM converged in %s iterations', epoch)
                return self.means, self.variances, self.mixture_coefficients

            self._maximization(data,
                               posterior_probabilities,
                               lr_means=lr_mean,
                               conv_factor_mean=conv_factor_mean)

        logging.info('WARNING: EM did not converge \n'
                     'Please increase MINIMUM_EPOCHS.')

        return self.means, self.variances, self.mixture_coefficients
