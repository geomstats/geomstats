"""Expectation maximization algorithm."""

import logging

import geomstats.backend as gs
from geomstats.geometry.poincare_ball \
    import Normalization_Factor_Storage, PoincareBall
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


class RiemannianEM():
    """Expectation-maximization class on Poincaré Ball.

    A class for performing Expectation-Maximization on the
    Poincaré Ball to fit data into a Gaussian Mixture Model.

    Parameters
    ----------
    riemannian_metric : object of class RiemannianMetric
        The geomstats Riemmanian metric associated with
                        the used manifold.
    n_gaussian : int
        Number of Gaussian components in the mix.
    initialisation_method : basestring
        Choice between initialization method for variances, means and weights.
           'random' : will select random uniformally train point as
                     initial centroids.
            # TODO: hzaatiti, tgeral68 implement kmeans initialisation
            'kmeans' : will apply Riemannian kmeans to deduce
            variances and means that the EM will use initially.
    tol : float
        Convergence factor. If the difference of mean distance
        between two step is lower than tol.
    mean_method: basestring
        Specify the method to compute the mean.
    point_type: basestring
        Specify whether to use vector or matrix representation.

    Returns
    -------
    self : object
        Returns the instance itself.
    """

    def __init__(self,
                 riemannian_metric,
                 n_gaussian=8,
                 initialisation_method='random',
                 tol=DEFAULT_TOL,
                 mean_method='default',
                 point_type='vector'):

        self.n_gaussian = n_gaussian
        self.riemannian_metric = riemannian_metric
        self.initialisation_method = initialisation_method
        self.tol = tol
        self.mean_method = mean_method
        self.point_type = point_type
        self._dimension = None
        self.mixture_coefficients = None
        self.variances = None
        self.means = None
        self.normalization_factor = None

    def update_posterior_probabilities(self, posterior_probabilities):
        """Posterior probabilities update function.

        Parameters
        ----------
        posterior_probabilities : array-like, shape=[n_samples, n_gaussian]
            Probability of a given sample to belong to a component
            of the GMM, computed for all components.
        """
        self.mixture_coefficients = gs.mean(posterior_probabilities, 0)

    def update_means(self, data, posterior_probabilities,
                     lr_means, tau_means, max_iter=DEFAULT_MAX_ITER):
        """Means update function."""
        n_gaussian = posterior_probabilities.shape[-1]

        mean = FrechetMean(
            metric=self.riemannian_metric,
            method=self.mean_method,
            lr=lr_means,
            tau=tau_means,
            max_iter=max_iter,
            point_type=self.point_type)

        data_expand = gs.expand_dims(data, 1)
        data_expand = gs.repeat(data_expand, n_gaussian, axis=1)

        mean.fit(data_expand, weights=posterior_probabilities)
        self.means = gs.squeeze(mean.estimate_)

    def update_variances(self, data, posterior_probabilities):
        """Variances update function.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        posterior_probabilities : array-like, shape=[n_samples, n_gaussian]
            Probability of a given sample to belong to a component
            of the GMM, computed for all components.
        """
        n_data, n_gaussian = data.shape[0], self.means.shape[0]

        data_expand = gs.expand_dims(data, 1)
        data_expand = gs.repeat(data_expand, n_gaussian, axis=1)
        means = gs.expand_dims(self.means, 0)
        means = gs.repeat(means, n_data, axis=0)

        data_flattened = gs.reshape(data_expand, (-1, data_expand.shape[-1]))
        means_flattened = gs.reshape(means, (-1, means.shape[-1]))

        dist_means_data = (self.riemannian_metric.dist(
            data_flattened, means_flattened) ** 2)

        dist_means_data = gs.reshape(dist_means_data,
                                     (data_expand.shape[0],
                                      data_expand.shape[1]))

        weighted_dist_means_data = (dist_means_data *
                                    posterior_probabilities).sum(0) / \
            posterior_probabilities.sum(0)

        self.variances = \
            self.normalization_factor.find_variance_from_index(
                weighted_dist_means_data)

    def _expectation(self, data):
        """Update the posterior probabilities.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        """
        probability_distribution_function = \
            PoincareBall.pdf(data,
                             self.means,
                             self.variances,
                             norm_func=self.
                             normalization_factor.
                             find_normalisation_factor,
                             metric=self.riemannian_metric)

        if (probability_distribution_function.mean() !=
                probability_distribution_function.mean()):
            logging.info('EXPECTATION : Probability distribution function'
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

            raise ValueError('EXPECTATION : posterior probabilities ' +
                             'contain elements that are not numbers.')

        if 1 - SUM_CHECK_PDF >= gs.mean(gs.sum(
                posterior_probabilities, 1)) >= 1 + SUM_CHECK_PDF:

            raise ValueError('EXPECTATION : posterior probabilities ' +
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
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        posterior_probabilities : array-like, shape=[n_samples, n_gaussian]
            Probability of a given sample to belong to a component
            of the GMM, computed for all components.
        lr_means : float
            Learning rate for computing the means.
        conv_factor_mean : float
            Convergence factor for means.
        max_iter : int
            Maximum number of iterations for computing the means.
        """
        self.update_posterior_probabilities(posterior_probabilities)

        if(gs.mean(self.mixture_coefficients)
                != gs.mean(self.mixture_coefficients)):
            raise NameError('UPDATE : mixture coefficients ' +
                            'contain elements that are not numbers')

        self.update_means(data,
                          posterior_probabilities,
                          lr_means=lr_means,
                          tau_means=conv_factor_mean,
                          max_iter=max_iter)

        if self.means.mean() != self.means.mean():
            raise NameError('UPDATE : means contain' +
                            'not a number elements')

        self.update_variances(data, posterior_probabilities)

        if self.variances.mean() != self.variances.mean():
            raise NameError('UPDATE : variances contain' +
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
            Maximum number of iterations.
        lr_mean : float
            Learning rate for the mean.
        conv_factor_mean : float
            Convergence factor for the mean.

        Returns
        -------
        self : object
            Return the components of the computed
            Gaussian mixture model: means, variances and mixture_coefficients.
        """
        self._dimension = data.shape[-1]
        self.means = (gs.random.rand(
            self.n_gaussian,
            self._dimension) - 0.5) / self._dimension
        self.variances = gs.random.rand(self.n_gaussian) / 10 + 0.8
        self.mixture_coefficients = \
            gs.ones(self.n_gaussian) / self.n_gaussian
        posterior_probabilities = gs.ones((data.shape[0],
                                           self.means.shape[0]))
        self.normalization_factor = Normalization_Factor_Storage(
            gs.arange(ZETA_LOWER_BOUND,
                      ZETA_UPPER_BOUND,
                      ZETA_STEP),
            self._dimension)

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
