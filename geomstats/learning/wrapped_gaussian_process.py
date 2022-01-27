r""" Wrapped Gaussian Process.

TODO

"""

import logging

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


class WrappedGaussianProcess(GaussianProcessRegressor):
    r""" Wrapped Gaussian Process.

    TODO

    Parameters
    ----------
    space : Manifold
        Manifold.
    metric : RiemannianMetric
        Riemannian metric.
    center_X : bool
        Subtract mean to X as a preprocessing.
    method : str, {\'extrinsic\', \'riemannian\'}
        Gradient descent method.
        Optional, default: extrinsic.
    max_iter : int
        Maximum number of iterations for gradient descent.
        Optional, default: 100.
    learning_rate : float
        Initial learning rate for gradient descent.
        Optional, default: 0.1
    tol : float
        Tolerance for loss minimization.
        Optional, default: 1e-5
    verbose : bool
        Verbose option.
        Optional, default: False.
    initialization : str or array-like,
        {'random', 'data', 'frechet', warm_start'}
        Initial values of the parameters for the optimization,
        or initialization method.
        Optional, default: 'random'
    regularization : float
        Weight on the constraint for the intercept to lie on the manifold in
        the extrinsic optimization scheme. An L^2 constraint is applied.
        Optional, default: 1.
    """

    def __init__(
            self,
            space,
            metric,
            prior,
            kernel=None,
            *,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            random_state=None,
    ):
        if metric is None:
            metric = space.metric
        self.metric = metric
        self.space = space
        self.prior = prior

        super(WrappedGaussianProcess, self).__init__(kernel=kernel,
                                                     alpha=alpha,
                                                     optimizer=optimizer,
                                                     n_restarts_optimizer=n_restarts_optimizer,
                                                     normalize_y=normalize_y,
                                                     copy_X_train=copy_X_train,
                                                     random_state=random_state)

    def _check_prior(self):
        """
        TODO
        """
        ...

        return True

    def _get_tangent_targets(self, X, y):
        """
        TODO
        """
        base_points = self.prior(X)
        return self.metric.log(y, base_point=base_points)

    def fit(self, X, y):
        """Fit Wrapped Gaussian process regression model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. The target must belongs to the manifold space
        Returns
        -------
        self : object
            WrappedGaussianProcessRegressor class instance.
        """
        assert self.space.belongs(y).all(), "The target values must belongs to the given space"
        tangent_y = self._get_tangent_targets(X, y)
        super(WrappedGaussianProcess, self).fit(X, tangent_y)

        return self

    def predict(self, X, return_std=False, return_cov=False):
        """Predict using the Gaussian process regression model.
        We can also predict based on an unfitted model by using the GP prior.
        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`) or covariance
        (`return_cov=True`). Note that at most one of the two can be requested.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated.
        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        return_cov : bool, default=False
            If True, the covariance of the joint predictive distribution at
            the query points is returned along with the mean.
        Returns
        -------
        y_mean : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Mean of predictive distribution a query points.
        y_std : ndarray of shape (n_samples,) or (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.
        y_cov : ndarray of shape (n_samples, n_samples) or \
                (n_samples, n_samples, n_targets), optional
            Covariance of joint predictive distribution a query points.
            Only returned when `return_cov` is True.
        """

        tangent_means = super(WrappedGaussianProcess, self).predict(X)
        base_points = self.prior(X)
        y_mean = self.metric.exp(tangent_means, base_point=base_points)
        return y_mean

    def sample_y(self, X, n_samples=1, random_state=0):
        """Draw samples from Gaussian process and evaluate at X.
        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features) or list of object
            Query points where the GP is evaluated.
        n_samples : int, default=1
            Number of samples drawn from the Gaussian process per query point.
        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.
            See :term:`Glossary <random_state>`.
        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from Gaussian process and
            evaluated at query points.
        """

        tangent_samples = super(WrappedGaussianProcess, self).sample_y(X, n_samples, random_state)
        y_samples = np.zeros(tangent_samples.shape)

        if len(tangent_samples.shape) == 2:
            base_points = self.prior(X)
            for i in range(tangent_samples.shape[1]):
                y_samples[:, i] = self.metric.exp(tangent_samples[:, i], base_point=base_points)
        else:  # len(tangent_samples.shape) == 3
            base_points = self.prior(X)
            for i in range(tangent_samples.shape[1]):
                y_samples[:, :, i] = self.metric.exp(tangent_samples[:, :, i], base_point=base_points)

        return y_samples
