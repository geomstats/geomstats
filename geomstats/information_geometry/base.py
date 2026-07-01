"""Mixin for manifolds of probability distributions."""

import math

import geomstats.backend as gs
from geomstats.geometry.connection import Connection
from geomstats.vectorization import get_batch_shape


class InformationManifoldMixin:
    """Mixin for manifolds of probability distributions."""

    def __init__(self, support_shape, **kwargs):
        self.support_shape = support_shape
        super().__init__(**kwargs)

    def sample(self, point, n_samples=1):
        """Sample from the probability distribution.

        Sample from the probability distribution with parameters provided
        by point. This gives n_samples points.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a probability distribution.
        n_samples : int
            Number of points to sample for each set of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from the probability distributions.
        """
        raise NotImplementedError("The sample method is not yet implemented.")

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the probability
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a probability distribution.

        Returns
        -------
        pdf : function
            Probability density function of the probability distribution with
            parameters provided by point.
        """
        raise NotImplementedError("`point_to_pdf` has not yet been implemented.")

    def point_to_cdf(self, point):
        """Compute cdf associated to point.

        Compute the cumulative density function of the probability
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a probability distribution.

        Returns
        -------
        cdf : function
            Cumulative density function of the probability distribution with
            parameters provided by point.
        """
        raise NotImplementedError("`point_to_cdf` has not yet been implemented.")


class ScipyRandomVariable:
    """A random variable."""

    def __init__(self, space, scp_rvs, scp_pdf=None):
        self.space = space
        self.scp_rvs = scp_rvs
        self.scp_pdf = scp_pdf


class ScipyUnivariateRandomVariable(ScipyRandomVariable):
    """A univariate random variable."""

    @staticmethod
    def _unflatten_res(flat_res, expected_shape):
        return gs.reshape(flat_res, expected_shape)

    def rvs(self, point, n_samples=1):
        """Sample from a univariate distribution.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point representing a univariate distribution.
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, *support_shape]
            Sample from distribution.
        """
        batch_shape = get_batch_shape(self.space.point_ndim, point)
        n_points = math.prod(batch_shape)

        pre_flat_shape = batch_shape + (n_samples,)
        params = self._flatten_params(point, pre_flat_shape)

        size = n_points * n_samples
        flat_sample = gs.from_numpy(self.scp_rvs(**params, size=size))

        expected_shape = batch_shape + (n_samples,) + self.space.support_shape
        return self._unflatten_res(flat_sample, expected_shape)

    def pdf(self, sample, point):
        """Evaluate the probability density function at x.

        Parameters
        ----------
        x : array-like, shape=[n_samples, *support_shape]
            Sample points at which to compute the probability density function.
        point : array-like, shape=[..., *space.shape]
            Point representing a distribution.

        Returns
        -------
        pdf_at_x : array-like, shape=[..., n_samples]
            Values of pdf at x for each value of the parameters provided
            by point.
        """
        batch_shape = get_batch_shape(self.space.point_ndim, point)
        n_points = math.prod(batch_shape)
        n_samples = sample.shape[0]

        pre_flat_shape = batch_shape + (n_samples,)
        params = self._flatten_params(point, pre_flat_shape)

        sample_ = gs.reshape(gs.broadcast_to(sample, (n_points, n_samples)), (-1,))

        pdf = gs.from_numpy(self.scp_pdf(sample_, **params))

        expected_shape = batch_shape + (n_samples,) + self.space.support_shape
        return self._unflatten_res(pdf, expected_shape)


class ScipyMultivariateRandomVariable(ScipyRandomVariable):
    """A multivariate random variable."""

    def _rvs_single(self, point, n_samples):
        return gs.to_ndarray(
            self.scp_rvs(point, size=n_samples),
            to_ndim=len(self.space.support_shape) + 1,
        )

    def rvs(self, point, n_samples=1):
        """Sample from a multivariate distribution.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point representing a distribution.
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, *support_shape]
            Sample from distribution.
        """
        if point.ndim > self.space.point_ndim:
            return gs.stack([self._rvs_single(point_, n_samples) for point_ in point])

        return self._rvs_single(point, n_samples)

    def _pdf_single(self, x, point):
        out = self.scp_pdf(x, point)
        if x.shape[0] == 1 and point.ndim == self.space.point_ndim:
            return gs.array([out])
        return gs.from_numpy(out)

    def pdf(self, x, point):
        """Evaluate the probability density function at x.

        Parameters
        ----------
        x : array-like, shape=[n_samples, *support_shape]
            Sample points at which to compute the probability density function.
        point : array-like, shape=[..., *space.shape]
            Point representing a distribution.

        Returns
        -------
        pdf_at_x : array-like, shape=[..., n_samples, *support_shape]
            Values of pdf at x for each value of the parameters provided
            by point.
        """
        if point.ndim > self.space.point_ndim:
            return gs.stack([self._pdf_single(x, point_) for point_ in point])

        return self._pdf_single(x, point)


class AlphaConnection(Connection):
    def __init__(self, space, riemannian_manifold, alpha=1.0):
        super().__init__(space)
        self._riemannian_manifold = riemannian_manifold
        self.alpha = alpha

    def first_kind_christoffels(self, base_point):
        r"""Compute the first kind Christoffel symbols.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            First kind Christoffel symbols.
        """
        raise NotImplementedError

    def christoffels(self, base_point):
        r"""Compute the (second kind) Christoffel symbols.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim]
            Second kind Christoffel symbols.
        """
        cometric_matrix = self._riemannian_manifold.metric.cometric_matrix(base_point)
        first_kind_christoffels = self.first_kind_christoffels(base_point)
        second_kind_christoffels = gs.einsum(
            "...kl, ...ijl -> ...kij", cometric_matrix, first_kind_christoffels
        )
        return second_kind_christoffels
    
    def jacobian_christoffels(self, base_point):
        r"""Compute the Jacobian of the Christoffel symbols.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.

        Returns
        -------
        matrix : array-like, shape=[..., dim, dim, dim, dim]
            Jacobian of the Christoffel symbols.
        """
        raise NotImplementedError
