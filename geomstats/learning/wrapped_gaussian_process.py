r"""Wrapped Gaussian Process.

Lead author: Arthur Pignet

Extension of Gaussian Processes to Riemannian Manifolds,
introduced in [Mallasto]_.

References
----------
..[Mallasto]   Mallasto, A. and Feragen, A.
            “Wrapped gaussian process
            regression on riemannian manifolds.”
            IEEE/CVF
Conference on Computer Vision and Pattern Recognition (2018)

"""

from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor

import geomstats.backend as gs


class WrappedGaussianProcess(MultiOutputMixin, RegressorMixin, BaseEstimator):
    r"""Wrapped Gaussian Process.

    The implementation is based on the algorithm 4 of [1].

    Parameters
    ----------
    space : Manifold
        Manifold.
    metric : RiemannianMetric
        Riemannian metric.
    prior : function
        Associate to each input a manifold valued point.
     kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel ``ConstantKernel(1.0, constant_value_bounds="fixed")
        * RBF(1.0, length_scale_bounds="fixed")`` is used as default. Note that
        the kernel hyperparameters are optimized during fitting unless the
        bounds are marked as "fixed".
    alpha : float or ndarray of shape (n_samples,), default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        This can prevent a potential numerical issue during fitting, by
        ensuring that the calculated values form a positive definite matrix.
        It can also be interpreted as the variance of additional Gaussian
        measurement noise on the training observations. Note that this is
        different from using a `WhiteKernel`. If an array is passed, it must
        have the same number of entries as the data used for fitting and is
        used as datapoint-dependent noise level. Allowing to specify the
        noise level directly as a parameter is mainly for convenience and
        for consistency with :class:`~sklearn.linear_model.Ridge`.
    optimizer : "fmin_l_bfgs_b" or callable, default="fmin_l_bfgs_b"
        Can either be one of the internally supported optimizers for optimizing
        the kernel's parameters, specified by a string, or an externally
        defined optimizer passed as a callable. If a callable is passed, it
        must have the signature::
            def optimizer(obj_func, initial_theta, bounds):
                # * 'obj_func': the objective function to be minimized, which
                #   takes the hyperparameters theta as a parameter and an
                #   optional flag eval_gradient, which determines if the
                #   gradient is returned additionally to the function value
                # * 'initial_theta': the initial value for theta, which can be
                #   used by local optimizers
                # * 'bounds': the bounds on the values of theta
                ....
                # Returned are the best found hyperparameters theta and
                # the corresponding value of the target function.
                return theta_opt, func_min
        Per default, the L-BFGS-B algorithm from `scipy.optimize.minimize`
        is used. If None is passed, the kernel's parameters are kept fixed.
        Available internal optimizers are: `{'fmin_l_bfgs_b'}`.
    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that `n_restarts_optimizer == 0` implies that one
        run is performed.
    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.

    [1] Mallasto, A. and Feragen, A. Wrapped gaussian process
    regression on riemannian manifolds. In 2018 IEEE/CVF
    Conference on Computer Vision and Pattern Recognition
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
        copy_X_train=True,
        random_state=None,
    ):
        if metric is None:
            metric = space.metric
        self.metric = metric
        self.space = space
        self.prior = prior
        self.copy_X_train = copy_X_train

        self.y_train_ = None
        self.tangent_y_train_ = None
        self.y_train_shape_ = None

        self._euclidean_gpr = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=False,
            copy_X_train=copy_X_train,
            random_state=random_state,
        )

        self.__dict__.update(self._euclidean_gpr.__dict__)
        self.log_marginal_likelihood = self._euclidean_gpr.log_marginal_likelihood

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
        return self.metric.log(y, base_point=base_points)

    def fit(self, X, y):
        """Fit Wrapped Gaussian process regression model.

        The Wrapped Gaussian process is fit through the following steps:

        - Compute the tangent dataset using the prior
        - Fit a Gaussian process regression on the tangent dataset
        - Store the resulting euclidean Gaussian process

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
        or (n_samples, n1_targets, n2_targets)
            Target values. The target must belongs to the manifold space

        Returns
        -------
        self : object
            WrappedGaussianProcessRegressor class instance.
        """
        if not gs.all(self.space.belongs(y)):
            raise AttributeError("The target values must belongs to the given space")

        # compute the tangent dataset using the prior
        tangent_y = self._get_tangent_targets(X, y)
        self.y_train_shape_ = y.shape[
            1:
        ]  # this is really useful when the samples are matrices, or tensor of dim>1
        tangent_y = gs.reshape(tangent_y, (y.shape[0], -1))  # flatten the samples.
        # fit a gpr on the tangent dataset

        self._euclidean_gpr.fit(X, tangent_y)
        # update the attributes of the wgpr using the new attributes of the gpr

        self.__dict__.update(self._euclidean_gpr.__dict__)
        self.y_train_ = y
        self.tangent_y_train_ = tangent_y  # = self._euclidean_gpr.y_train_

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
        if return_tangent_cov:
            tangent_means, tangent_cov = self._euclidean_gpr.predict(
                X, return_cov=True, return_std=False
            )
            tangent_means = gs.reshape(
                gs.cast(tangent_means, dtype=X.dtype),
                (X.shape[0], *self.y_train_shape_),
            )
            tangent_cov = gs.cast(
                tangent_cov, dtype=X.dtype
            )  # covariance of the vectorized predictions.

            base_points = self.prior(X)
            y_mean = self.metric.exp(tangent_means, base_point=base_points)
            result = (y_mean, tangent_cov)

        elif return_tangent_std:
            tangent_means, tangent_std = self._euclidean_gpr.predict(
                X, return_cov=False, return_std=True
            )
            base_points = self.prior(X)
            tangent_means = gs.reshape(
                gs.cast(tangent_means, dtype=X.dtype),
                (X.shape[0], *self.y_train_shape_),
            )
            tangent_std = gs.cast(tangent_std, dtype=X.dtype)

            y_mean = self.metric.exp(tangent_means, base_point=base_points)
            result = (y_mean, tangent_std)

        else:
            tangent_means = self._euclidean_gpr.predict(
                X, return_cov=False, return_std=False
            )
            base_points = self.prior(X)
            tangent_means = gs.reshape(
                gs.cast(tangent_means, dtype=X.dtype),
                (X.shape[0], *self.y_train_shape_),
            )
            y_mean = self.metric.exp(tangent_means, base_point=base_points)
            result = y_mean

        return result

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
            Number of samples drawn from the Wrapped Gaussian process per query point.
        random_state : int, RandomState instance or None, default=0
            Determines random number generation to randomly draw samples.
            Pass an int for reproducible results across multiple function
            calls.

        Returns
        -------
        y_samples : ndarray of shape (n_samples_X, n_samples), or \
            (n_samples_X, n_targets, n_samples)
            Values of n_samples samples drawn from wrapped Gaussian process and
            evaluated at query points.
        """
        tangent_samples = self._euclidean_gpr.sample_y(X, n_samples, random_state)
        tangent_samples = gs.cast(tangent_samples, dtype=X.dtype)
        # flatten the samples
        tangent_samples = gs.reshape(
            gs.transpose(tangent_samples, [0, 2, 1]), (-1, *self.y_train_shape_)
        )

        # generate the base_points
        base_points = self.prior(X)
        # repeat the base points in order to match the tangent samples
        base_points = gs.repeat(gs.expand_dims(base_points, 2), n_samples, axis=2)
        # flatten the base_points
        base_points = gs.reshape(
            gs.transpose(base_points, [0, 2, 1]), (-1, *self.y_train_shape_)
        )

        # get the flattened samples
        y_samples = self.metric.exp(tangent_samples, base_point=base_points)
        y_samples = gs.transpose(
            gs.reshape(y_samples, (X.shape[0], n_samples, *self.y_train_shape_)),
            [0, 2, 1],
        )

        return y_samples
