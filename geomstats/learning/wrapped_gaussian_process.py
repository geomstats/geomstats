r"""Wrapped Gaussian Process.

Lead author: Arthur Pignet

Extension of Gaussian Processes to Riemannian Manifolds,
introduced in [Mallasto]_.

References
----------
.. [Mallasto] Mallasto, A. and Feragen, A.
    “Wrapped gaussian process regression on riemannian manifolds.”
    IEEE/CVF Conference on Computer Vision and Pattern Recognition
    (2018)

"""

from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor

import geomstats.backend as gs


class WrappedGaussianProcess(MultiOutputMixin, RegressorMixin, BaseEstimator):
    r"""Wrapped Gaussian Process.

    The implementation is based on the algorithm 4 of [1]_.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    prior : callable
        Associate to each input a manifold valued point.

    References
    ----------
    .. [1] Mallasto, A. and Feragen, A. Wrapped gaussian process
        regression on riemannian manifolds. In 2018 IEEE/CVF
        Conference on Computer Vision and Pattern Recognition
    """

    def __init__(self, space, prior):
        self.space = space
        self.prior = prior

        self.euclidean_gpr = GaussianProcessRegressor(
            kernel=None,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            random_state=None,
        )

        self.tangent_y_train_ = None

    def set(self, **kwargs):
        """Set euclidean_gpr parameters.

        Especially useful for one line instantiations.
        """
        for param_name, value in kwargs.items():
            if not hasattr(self.euclidean_gpr, param_name):
                raise ValueError(f"Unknown parameter {param_name}.")

            setattr(self.euclidean_gpr, param_name, value)
        return self

    def _get_tangent_targets(self, X, y):
        """Compute the tangent targets, using the provided prior.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
        or (n_samples, n1_targets, n2_targets) for
        matrix-valued targets.
            Target values. The target must belongs to the manifold space

        Returns
        -------
        tangent_y : array-like of shape (n_samples,) or (n_samples, n_targets)
        or (n_samples, n1_targets, n2_targets)
                Target projected on the associated (by the prior) tangent space.
        """
        base_points = self.prior(X)
        return self.space.metric.log(y, base_point=base_points)

    def fit(self, X, y):
        """Fit Wrapped Gaussian process regression model.

        The Wrapped Gaussian process is fit through the following steps:

        - Compute the tangent dataset using the prior
        - Fit a Gaussian process regression on the tangent dataset
        - Store the resulting euclidean Gaussian process

        Parameters
        ----------
        X : array-like, shape=[n_samples,]
            Training input samples.
        y : array-like, shape[n_samples, {dim, [n,n]}]
            Training target values.

        Returns
        -------
        self : object
            Returns self.
        """
        self.tangent_y_train_ = tangent_y = self._get_tangent_targets(X, y)
        tangent_y = gs.reshape(tangent_y, (y.shape[0], -1))

        self.euclidean_gpr.fit(X, tangent_y)

        return self

    def predict(self, X, return_tangent_std=False, return_tangent_cov=False):
        """Predict using the Gaussian process regression model.

        A fitted Wrapped Gaussian process can be use to predict values
        through the following steps:

        - Use the stored Gaussian process regression on the dataset to
          return tangent predictions
        - Compute the base-points using the prior
        - Map the tangent predictions on the manifold via the metric's exp
          with the base-points yielded by the prior

        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.
        return_tangent_std : bool, default=False
            If True, the standard-deviation of the predictive distribution on at
            the query points in the tangent space is returned along with the mean.
        return_tangent_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points in the tangent space is returned along with the mean.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.
        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points in
            the tangent space.
            Only returned when `return_std` is True.
        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points
            in the tangent space.
            Only returned when `return_cov` is True.
            In the case where the target is matrix valued,
            return the covariance of the vectorized prediction.
        """
        euc_result = self.euclidean_gpr.predict(
            X, return_cov=return_tangent_cov, return_std=return_tangent_std
        )

        return_multiple = return_tangent_std or return_tangent_cov
        tangent_means = euc_result[0] if return_multiple else euc_result

        base_points = self.prior(X)
        tangent_means = gs.reshape(
            gs.from_numpy(tangent_means),
            (X.shape[0], *self.space.shape),
        )
        y_mean = self.space.metric.exp(tangent_means, base_point=base_points)

        if return_multiple:
            tangent_std_cov = gs.from_numpy(euc_result[1])
            return (y_mean, tangent_std_cov)

        return y_mean

    def sample_y(self, X, n_samples=1, random_state=0):
        """Draw samples from Wrapped Gaussian process and evaluate at X.

        A fitted Wrapped Gaussian process can be use to sample
        values through the following steps:

        - Use the stored Gaussian process regression on the dataset
          to sample tangent values
        - Compute the base-points using the prior
        - Flatten (and repeat if needed) both the base-points and the
          tangent samples to benefit from vectorized computation.
        - Map the tangent samples on the manifold via the metric's exp with the
          flattened and repeated base-points yielded by the prior

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the WGP is evaluated.
        n_samples : int, default=1
            Number of samples drawn from the Wrapped Gaussian process per query
            point.
        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, *target_shape, n_samples)
            Values of n_samples samples drawn from wrapped Gaussian process and
            evaluated at query points.
        """
        tangent_samples = gs.from_numpy(
            self.euclidean_gpr.sample_y(X, n_samples, random_state)
        )

        if gs.ndim(tangent_samples) > 2:
            tangent_samples = gs.moveaxis(tangent_samples, -2, -1)

        flat_tangent_samples = gs.reshape(tangent_samples, (-1, *self.space.shape))

        base_points = gs.repeat(self.prior(X), n_samples, axis=0)

        flat_y_samples = self.space.metric.exp(
            flat_tangent_samples, base_point=base_points
        )

        y_samples = gs.reshape(
            flat_y_samples, (X.shape[0], n_samples, *self.space.shape)
        )

        if gs.ndim(tangent_samples) > 2:
            y_samples = gs.moveaxis(y_samples, 1, -1)

        return y_samples
