"""Expectation maximization algorithm.

Lead authors: Thomas Gerald and Hadi Zaatiti.
"""

import logging

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin
from geomstats.learning.frechet_mean import FrechetMean, variance
from geomstats.learning.kmeans import RiemannianKMeans

PDF_TOL = 1e-6
SUM_CHECK_PDF = 1e-4
MIN_VAR_INIT = 1e-3


class GaussianMixtureModel:
    r"""Gaussian mixture model (GMM).

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    means : array-like, shape=[n_gaussians, dim]
        Means of each component of the GMM.
    variances : array-like, shape=[n_gaussians,]
        Variances of each component of the GMM.

    Attributes
    ----------
    normalization_factor_var : array-like, shape=[n_variances,]
        Array of computed normalization factor.
    variances_range : array-like, shape=[n_variances,]
        Array of standard deviations.
    phi_inv_var : array-like, shape=[n_variances,]
        Array of the computed inverse of a function phi
        whose expression is closed-form
        :math:`\sigma\mapsto \sigma^3 \times \frac{d}
        {\mathstrut d\sigma}\log \zeta_m(\sigma)`
        where :math:`\sigma` denotes the variance
        and :math:`\zeta` the normalization coefficient
        and :math:`m` the dimension.
    """

    def __init__(
        self,
        space,
        means=None,
        variances=None,
        zeta_lower_bound=5e-1,
        zeta_upper_bound=2.0,
        zeta_step=0.01,
    ):
        self.space = space
        self.means = means
        self.variances = variances

        self.zeta_lower_bound = zeta_lower_bound
        self.zeta_upper_bound = zeta_upper_bound
        self.zeta_step = zeta_step

        (
            self.variances_range,
            self.normalization_factor_var,
            self.phi_inv_var,
        ) = self._normalization_factor_init()

    def _normalization_factor_init(self):
        r"""Set up function for the normalization factor.

        The normalization factor is used to define Gaussian distributions
        at initialization.
        """
        variances = gs.arange(
            self.zeta_lower_bound, self.zeta_upper_bound, self.zeta_step
        )
        normalization_factor_var = self.space.metric.normalization_factor(variances)

        cond_1 = normalization_factor_var.sum() != normalization_factor_var.sum()
        cond_2 = normalization_factor_var.sum() == float("+inf")
        cond_3 = normalization_factor_var.sum() == float("-inf")

        if cond_1 or cond_2 or cond_3:
            logging.warning("Untractable normalization factor :")

            limit_nf = (
                ((normalization_factor_var / normalization_factor_var) * 0)
                .nonzero()[0]
                .item()
            )
            max_nf = len(variances)
            variances = variances[0:limit_nf]
            normalization_factor_var = normalization_factor_var[0:limit_nf]
            if cond_1:
                logging.warning("\t Nan value " "in processing normalization factor")
            if cond_2 or cond_3:
                raise ValueError("\t +-inf value in " "processing normalization factor")

            logging.warning("\t Max variance is now : %s", str(variances[-1]))
            logging.warning(
                "\t Number of possible variance is now: %s / %s ",
                str(len(variances)),
                str(max_nf),
            )

        _, log_grad_zeta = self.space.metric.norm_factor_gradient(variances)

        phi_inv_var = variances**3 * log_grad_zeta

        return variances, normalization_factor_var, phi_inv_var

    def pdf(self, data):
        """Return the separate probability density function of GMM.

        The probability density function is computed for
        each component of the GMM separately (i.e., mixture coefficients
        are not taken into account).

        Parameters
        ----------
        data : array-like, shape=[n_samples, dim]
            Points at which the GMM probability density is computed.

        Returns
        -------
        pdf : array-like, shape=[n_samples, n_gaussians,]
            Probability density function computed at each data
            sample and for each component of the GMM.
        """
        data_length, _, _ = data.shape + (self.means.shape[0],)

        variances_expanded = gs.expand_dims(self.variances, 0)
        variances_expanded = gs.repeat(variances_expanded, data_length, 0)

        variances_flatten = variances_expanded.flatten()

        distances = -(self.space.metric.dist_broadcast(data, self.means) ** 2)
        distances = gs.reshape(distances, (data.shape[0] * self.variances.shape[0],))

        num = gs.exp(distances / (2 * variances_flatten**2))

        den = self._compute_normalization_factor()

        den = gs.expand_dims(den, 0)
        den = gs.repeat(den, data_length, axis=0).flatten()

        pdf = num / den
        pdf = gs.reshape(pdf, (data.shape[0], self.means.shape[0]))

        return pdf

    def _compute_normalization_factor(self):
        """Find the normalization factor given some variances.

        Returns
        -------
        norm_factor : array-like, shape=[n_gaussians,]
            Array of normalization factors for the given
            variances.
        """
        n_gaussians, precision = self.variances.shape[0], self.variances_range.shape[0]

        ref = gs.expand_dims(self.variances_range, 0)
        ref = gs.repeat(ref, n_gaussians, axis=0)
        val = gs.expand_dims(self.variances, 1)
        val = gs.repeat(val, precision, axis=1)

        difference = gs.abs(ref - val)

        index = gs.argmin(difference, axis=-1)
        norm_factor = self.normalization_factor_var[index]

        return norm_factor

    def compute_variance_from_index(self, weighted_distances):
        r"""Return the variance given weighted distances.

        Parameters
        ----------
        weighted_distances : array-like, shape=[n_gaussians,]
            Mean of the weighted distances between training data
            and current barycentres. The weights of each data sample
            corresponds to the probability of belonging to a component
            of the Gaussian mixture model.

        Returns
        -------
        var : array-like, shape=[n_gaussians,]
            Estimated variances for each component of the GMM.
        """
        n_gaussians, precision = (
            weighted_distances.shape[0],
            self.variances_range.shape[0],
        )

        ref = gs.expand_dims(self.phi_inv_var, 0)
        ref = gs.repeat(ref, n_gaussians, axis=0)

        val = gs.expand_dims(weighted_distances, 1)
        val = gs.repeat(val, precision, axis=1)

        abs_difference = gs.abs(ref - val)

        index = gs.argmin(abs_difference, -1)

        var = self.variances_range[index]

        return var

    def weighted_pdf(self, mixture_coefficients, mesh_data):
        """Return the probability density function of a GMM.

        Parameters
        ----------
        mixture_coefficients : array-like, shape=[n_gaussians,]
            Coefficients of the Gaussian mixture model.
        mesh_data : array-like, shape=[n_precision, dim]
            Points at which the GMM probability density is computed.

        Returns
        -------
        weighted_pdf : array-like, shape=[n_precision, n_gaussians,]
            Probability density function computed for each point of
            the mesh data, for each component of the GMM.
        """
        distance_to_mean = self.space.metric.dist_broadcast(mesh_data, self.means)

        variances_units = gs.expand_dims(self.variances, 0)
        variances_units = gs.repeat(variances_units, distance_to_mean.shape[0], axis=0)

        distribution_normal = gs.exp(
            -(distance_to_mean**2) / (2 * variances_units**2)
        )

        zeta_sigma = (2 * gs.pi) ** (2 / 3) * self.variances
        zeta_sigma = zeta_sigma * gs.exp(
            (self.variances**2 / 2) * gs.erf(self.variances / gs.sqrt(2))
        )

        result_num = gs.expand_dims(mixture_coefficients, 0)
        result_num = gs.repeat(result_num, len(distribution_normal), axis=0)
        result_num = result_num * distribution_normal

        result_denum = gs.expand_dims(zeta_sigma, 0)
        result_denum = gs.repeat(result_denum, len(distribution_normal), axis=0)

        weighted_pdf = result_num / result_denum

        return weighted_pdf


class RiemannianEM(TransformerMixin, ClusterMixin, BaseEstimator):
    r"""Expectation-maximization algorithm.

    A class for performing Expectation-Maximization to fit a Gaussian Mixture
    Model (GMM) to data on a manifold. This method is only implemented for
    the hypersphere and the Poincare ball.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    n_gaussians : int
        Number of Gaussian components in the mix.
    initialisation_method : basestring
        Optional, default: 'random'.
        Choice between initialization method for variances, means and weights.

        - 'random' : will select random uniformly train points as
          initial cluster centers.
        - 'kmeans' : will apply Riemannian kmeans to deduce
          variances and means that the EM will use initially.
    tol : float
        Optional, default: 1e-2.
        Convergence tolerance. If the difference of mean distance
        between two steps is lower than tol.
    max_iter : int
        Maximum number of iterations for the gradient descent.
        Optional, default: 100.

    Attributes
    ----------
    mixture_coefficients_ : array-like, shape=[n_gaussians,]
        Weights for each GMM component.
    variances_ : array-like, shape=[n_gaussians,]
        Variances for each GMM component.
    means_ : array-like, shape=[n_gaussian, _dimension]
        Barycentre of each component of the GMM.

    Example
    -------
    Available example on the Poincaré Ball manifold
    :mod:`examples.plot_expectation_maximization_ball`
    """

    def __init__(
        self,
        space,
        n_gaussians=8,
        initialisation_method="random",
        tol=1e-2,
        max_iter=100,
        conv_rate=1e-4,
        minimum_epochs=10,
    ):
        self.space = space
        self.n_gaussians = n_gaussians
        self.initialisation_method = initialisation_method
        self.tol = tol
        self.max_iter = max_iter
        self.conv_rate = conv_rate
        self.minimum_epochs = minimum_epochs

        self.mean_estimator = FrechetMean(space)
        if isinstance(self.mean_estimator, FrechetMean):
            self.mean_estimator.method = "batch"
            self.mean_estimator.set(
                max_iter=100,
                epsilon=1e-4,
                init_step_size=1.0,
            )

        self._model = GaussianMixtureModel(self.space)

        self.mixture_coefficients_ = None

    @property
    def means_(self):
        """Means of each component of the GMM."""
        return self._model.means

    @property
    def variances_(self):
        """Array of standard deviations."""
        return self._model.variances

    def _update_posterior_probabilities(self, posterior_probabilities):
        """Posterior probabilities update function.

        Parameters
        ----------
        posterior_probabilities : array-like, shape=[n_samples, n_gaussians,]
            Probability of a given sample to belong to a component
            of the GMM, computed for all components.
        """
        self.mixture_coefficients_ = gs.mean(posterior_probabilities, 0)

        if gs.any(gs.isnan(self.mixture_coefficients_)):
            logging.warning(
                "UPDATE : mixture coefficients contain elements that are not numbers"
            )

    def _update_means(self, data, posterior_probabilities):
        """Update means."""
        n_gaussians = posterior_probabilities.shape[-1]

        data_expand = gs.expand_dims(data, 1)
        data_expand = gs.repeat(data_expand, n_gaussians, axis=1)

        self.mean_estimator.fit(data_expand, weights=posterior_probabilities)
        self._model.means = gs.squeeze(self.mean_estimator.estimate_)

        if gs.any(gs.isnan(self._model.means)):
            logging.warning("UPDATE : means contain not a number elements")

    def _update_variances(self, data, posterior_probabilities):
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
        dist_means_data = self.space.metric.dist_broadcast(data, self._model.means) ** 2

        weighted_dist_means_data = (dist_means_data * posterior_probabilities).sum(
            0
        ) / posterior_probabilities.sum(0)

        self._model.variances = self._model.compute_variance_from_index(
            weighted_dist_means_data
        )

        if gs.any(gs.isnan(self._model.variances)):
            logging.warning("UPDATE : variances contain not a number elements")

    def _expectation(self, data):
        """Update the posterior probabilities.

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        """
        pdf = self._model.pdf(data)

        if gs.any(gs.isnan(pdf)):
            logging.warning(
                "EXPECTATION : Probability distribution function"
                "contain elements that are not numbers"
            )

        num_normalized_pdf = gs.einsum("j,...j->...j", self.mixture_coefficients_, pdf)
        valid_pdf_condition = gs.amin(gs.sum(num_normalized_pdf, -1))

        if valid_pdf_condition <= PDF_TOL:
            num_normalized_pdf[gs.sum(num_normalized_pdf, -1) <= PDF_TOL] = 1

        sum_pdf = gs.sum(num_normalized_pdf, -1)
        posterior_probabilities = gs.einsum(
            "...i,...->...i", num_normalized_pdf, 1 / sum_pdf
        )

        if gs.any(gs.mean(posterior_probabilities)) is None:
            logging.warning(
                "EXPECTATION : posterior probabilities "
                "contain elements that are not numbers."
            )

        if (
            1 - SUM_CHECK_PDF
            >= gs.mean(gs.sum(posterior_probabilities, 1))
            >= 1 + SUM_CHECK_PDF
        ):
            logging.warning("EXPECTATION : posterior probabilities " "do not sum to 1.")

        if gs.any(gs.sum(posterior_probabilities, 0) < PDF_TOL):
            logging.warning(
                "EXPECTATION : Gaussian got no elements "
                "(precision error) reinitialize"
            )
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
        self._update_posterior_probabilities(posterior_probabilities)
        self._update_means(data, posterior_probabilities)
        self._update_variances(data, posterior_probabilities)

    def _initialization(self, X):
        if self.initialisation_method == "kmeans":
            kmeans_estimator = RiemannianKMeans(
                space=self.space,
                n_clusters=self.n_gaussians,
                init="random",
            )

            kmeans_estimator.fit(X=X)
            cluster_centers = kmeans_estimator.cluster_centers_
            labels = kmeans_estimator.labels_

            means = cluster_centers
            variances = gs.zeros(self.n_gaussians)

            labeled_data = gs.vstack([labels, gs.transpose(X)])
            labeled_data = gs.transpose(labeled_data)
            for label, cluster_center in enumerate(cluster_centers):
                label_mask = gs.where(labeled_data[:, 0] == label)
                grouped_by_label = labeled_data[label_mask][:, 1:]
                v = variance(self.space, grouped_by_label, cluster_center)
                if grouped_by_label.shape[0] == 1:
                    v += MIN_VAR_INIT
                variances[label] = v
        else:
            dim = self.space.shape[-1]
            means = (gs.random.rand(self.n_gaussians, dim) - 0.5) / dim
            variances = gs.random.rand(self.n_gaussians) / 10 + 0.8

        return means, variances

    def fit(self, X, y=None):
        """Fit a Gaussian mixture model (GMM) given the data.

        Alternates between Expectation and Maximization steps
        for some number of iterations.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : None
            Target values. Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        self._model.means, self._model.variances = self._initialization(X)

        self.mixture_coefficients_ = gs.ones(self.n_gaussians) / self.n_gaussians
        posterior_probabilities = gs.ones((X.shape[0], self.n_gaussians))

        for epoch in range(self.max_iter):
            old_posterior_probabilities = posterior_probabilities

            posterior_probabilities = self._expectation(X)

            condition = gs.mean(
                gs.abs(old_posterior_probabilities - posterior_probabilities)
            )

            if condition < self.conv_rate and epoch > self.minimum_epochs:
                logging.info("EM converged in %s iterations", epoch)
                break

            self._maximization(X, posterior_probabilities)
        else:
            logging.info(
                "WARNING: EM did not converge \nPlease increase MINIMUM_EPOCHS."
            )

        return self
