"""Module for function spaces as geometric objects."""

import math

import geomstats.backend as gs
from geomstats._einsum import scalarvecmul
from geomstats.geometry.euclidean import CoordinateScalingDiffeo
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric


class HilbertSphere(Manifold):
    """Class for space of one dimensional functions with norm 1.

    The tangent space is given by functions that have
    zero inner-product with the base point

    Parameters
    ----------
    domain : array-like, shape=[n_samples,]
        Grid points on the domain.

    References
    ----------
    .. [Sea2016] Srivastava, Anuj, and Eric P. Klassen.
        Functional and shape data analysis.
        Vol. 1. New York: Springer, 2016.
    """

    def __init__(self, domain, equip=True):
        self.domain = domain
        super().__init__(
            dim=math.inf,
            shape=(len(domain),),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return HilbertSphereMetric

    def projection(self, point):
        """Project a point to the infinite dimensional hypersphere.

        Parameters
        ----------
        point : array-like, shape=[..., n_samples]
            Discrete evaluation of a function.

        Returns
        -------
        projected_point : array-like, shape=[..., n_samples]
            Point projected to the hypersphere.
        """
        norm = self.metric.norm(point, point)
        return scalarvecmul(1.0 / norm, point)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the vector space.

        This method checks the shape of the input point.

        Parameters
        ----------
        point : array-like, shape=[..., n_samples]
            Point to test.
        atol : float

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        norms = self.metric.norm(point, point)

        return gs.isclose(norms, 1.0, atol)

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., n_samples]
            Vector.
        base_point : array-like, shape=[..., n_samples]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        inner_product = self.metric.inner_product(vector, base_point, base_point)

        return gs.isclose(inner_product, gs.zeros_like(inner_product), atol)

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        Parameters
        ----------
        vector : array-like, shape=[..., n_samples]
            Vector.
        base_point : array-like, shape=[..., n_samples]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n_samples]
            Tangent vector at base point.
        """
        sq_norm = self.metric.squared_norm(base_point, base_point)
        inner_product = self.metric.inner_product(vector, base_point, base_point)
        coef = inner_product / sq_norm
        return vector - scalarvecmul(coef, base_point)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random points on the manifold.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            This parameter is ignored

        Returns
        -------
        samples : array-like, shape=[..., dim]
        """
        n_domain = len(self.domain)
        shape = (
            (n_domain,)
            if n_samples == 1
            else (
                n_samples,
                n_domain,
            )
        )
        point = gs.random.rand(*shape)

        return self.projection(point)


def _trapezoid_weights(grid):
    """Get the weights of the trapezoidal rule on a grid.

    Parameters
    ----------
    grid : array-like, shape=[n_samples,]
        Increasing grid points.

    Returns
    -------
    weights : array-like, shape=[n_samples,]
        Weights.
    """
    steps = grid[1:] - grid[:-1]
    zero = gs.zeros(1, dtype=steps.dtype)
    return (gs.concatenate([steps, zero]) + gs.concatenate([zero, steps])) / 2


class HilbertSphereMetric(PullbackDiffeoMetric):
    r"""A Riemannian metric on the Hilbert sphere (functions of norm 1).

    The inner product of functions is approximated with the trapezoidal rule,
    on the domain rescaled to :math:`[0, 1]`, i.e. by
    :math:`\langle f, g \rangle_w = \sum_i w_i f_i g_i`, with :math:`w` the
    trapezoidal weights. Scaling the samples by :math:`\sqrt{w}` sends it to
    the dot product, so that the metric is the pullback of the metric of the
    unit sphere by this scaling.
    """

    def __init__(self, space):
        domain = space.domain
        rescaled_domain = (domain - min(domain)) / (max(domain) - min(domain))
        scales = gs.sqrt(_trapezoid_weights(rescaled_domain))
        super().__init__(
            space,
            diffeo=CoordinateScalingDiffeo(scales),
            image_space=Hypersphere(dim=len(domain) - 1),
        )
