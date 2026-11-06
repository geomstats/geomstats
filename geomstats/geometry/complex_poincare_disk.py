"""The complex Poincaré disk manifold.

The Poincaré complex disk is a representation of the Hyperbolic space of dimension 2.

Lead author: Yann Cabanes.

References
----------
.. [Cabanes2022] Yann Cabanes. Multidimensional complex stationary
    centered Gaussian autoregressive time series machine learning
    in Poincaré and Siegel disks: application for audio and radar
    clutter classification, PhD thesis, 2022
.. [JV2016] B. Jeuris and R. Vandebril. The Kahler mean of Block-Toeplitz
    matrices with Toeplitz structured blocks, 2016.
    https://epubs.siam.org/doi/pdf/10.1137/15M102112X
.. [Yang2013] Marc Arnaudon, Frédéric Barbaresco and Le Yang.
    Riemannian medians and means with applications to radar signal processing,
    IEEE Journal of Selected Topics in Signal Processing, vol. 7, no. 4, pp. 595-604,
    Aug. 2013, doi: 10.1109/JSTSP.2013.2261798.
    https://ieeexplore.ieee.org/document/6514112
"""

import math

import geomstats.backend as gs
from geomstats.geometry.base import ComplexVectorSpaceOpenSet
from geomstats.geometry.complex_riemannian_metric import ComplexRiemannianMetric
from geomstats.geometry.hermitian import Hermitian
from geomstats.vectorization import repeat_out


class ComplexPoincareDisk(ComplexVectorSpaceOpenSet):
    """Class for the complex Poincaré disk.

    The Poincaré disk is a representation of the Hyperbolic
    space of dimension 2. Its complex dimension is 1.
    """

    def __init__(self, equip=True):
        super().__init__(
            dim=1,
            embedding_space=Hermitian(dim=1),
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ComplexPoincareDiskMetric

    @staticmethod
    def belongs(point, atol=gs.atol):
        """Check if a point belongs to the complex unit disk.

        Check if a point belongs to the Poincaré complex disk,
        i.e. Check if its norm is lower than one.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point to be checked.
        atol : float
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[...]
            Boolean denoting if point belongs to the
            complex Poincaré disk.
        """
        if point.shape[-1] != 1:
            return gs.zeros(point.shape[:-1], dtype=bool)

        return gs.all(gs.abs(point) < 1, axis=-1)

    @staticmethod
    def projection(point):
        """Project a point on the complex unit disk.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point to project.
        atol : float
            Tolerance.

        Returns
        -------
        projected : array-like, shape=[..., 1]
            Projected point.
        """
        scalars = gs.where(
            gs.abs(point) >= 1 - gs.atol,
            (1 - gs.atol) / gs.abs(point),
            1.0,
        )
        return scalars * point

    @staticmethod
    def random_point(n_samples=1, bound=0.8):
        """Generate random points in the complex unit disk.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Bound of the modulus of the sampled points.

        Returns
        -------
        samples : array-like, shape=[..., 1]
            Points sampled in the unit disk.
        """
        size = (n_samples, 1) if n_samples != 1 else (1,)
        modulus = bound * gs.random.rand(*size, dtype=gs.get_default_cdtype())
        angle = 2 * gs.pi * gs.random.rand(*size, dtype=gs.get_default_cdtype())
        return modulus * gs.exp(1j * angle)

    def random_tangent_vec(self, base_point, n_samples=1):
        r"""Sample tangent vectors at base point.

        The samples are scaled by the inverse conformal factor
        :math:`1 - |x|^2`, so that their norm for the Poincaré metric does not
        depend on the base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.
        n_samples : int
            Number of samples.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vectors at base point.
        """
        tangent_vec = super().random_tangent_vec(base_point, n_samples=n_samples)
        return (1 - gs.abs(base_point) ** 2) * tangent_vec


class ComplexPoincareDiskMetric(ComplexRiemannianMetric):
    """Class for the complex Poincaré metric."""

    def metric_matrix(self, base_point):
        """Compute the metric matrix at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        inner_prod_mat : array-like, shape=[...]
            Inner product matrix.
        """
        inner_prod_mat = 1 / (1 - gs.abs(base_point) ** 2) ** 2
        return gs.expand_dims(inner_prod_mat, axis=-1)

    @staticmethod
    def exp(tangent_vec, base_point):
        """Compute the complex Poincaré disk exponential map.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., 1]
            Riemannian exponential.
        """
        base_point, tangent_vec = gs.broadcast_arrays(base_point, tangent_vec)

        theta = gs.angle(tangent_vec)
        s = 2 * gs.abs(tangent_vec) / (1 - gs.abs(base_point) ** 2)

        exp_i_theta = gs.exp(1j * theta)
        exp_minus_s = gs.exp(-s)

        num = base_point + exp_i_theta + (base_point - exp_i_theta) * exp_minus_s
        den = (
            1
            + gs.conj(base_point) * exp_i_theta
            + (1 - gs.conj(base_point) * exp_i_theta) * exp_minus_s
        )
        return num / den

    @staticmethod
    def _tau(point_a, point_b, atol=gs.atol):
        """Compute the coefficient tau.

        The coefficient tau is used to compute the distance
        between point_a and point_b; it is also used to
        compute the logarithm map between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., 1]
            Point.
        point_b : array-like, shape=[..., 1]
            Point.
        atol : float
            Tolerance.

        Returns
        -------
        tau : array-like, shape=[...]
            Coefficient tau.
        """
        num = gs.abs(point_a - point_b)
        den = gs.maximum(gs.abs(1 - point_a * gs.conj(point_b)), atol)
        delta = gs.squeeze(gs.minimum(num / den, 1 - atol), axis=-1)
        return (1 / 2) * gs.log((1 + delta) / (1 - delta))

    def log(self, point, base_point):
        """Compute the complex Poincaré disk logarithm map.

        Compute the Riemannian logarithm at point base_point,
        of point wrt the positive reals metric.
        This gives a tangent vector at point base_point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        log : array-like, shape=[..., 1]
            Riemannian logarithm.
        """
        angle = gs.angle(point - base_point) - gs.angle(1 - gs.conj(base_point) * point)
        return gs.exp(1j * angle) * gs.einsum(
            "...,...i->...i",
            self._tau(base_point, point),
            (1 - gs.abs(base_point) ** 2),
        )

    def squared_dist(self, point_a, point_b, atol=gs.atol):
        """Compute the complex Poincaré disk squared distance.

        Compute the Riemannian squared distance between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., 1]
            Point.
        point_b : array-like, shape=[..., 1]
            Point.

        Returns
        -------
        squared_dist : array-like, shape=[...]
            Riemannian squared distance.
        """
        dist = self._tau(point_a, point_b, atol=atol)
        return gs.power(dist, 2)

    def dist(self, point_a, point_b, atol=gs.atol):
        """Compute the complex Poincaré disk distance.

        Compute the Riemannian distance between point_a and point_b.

        Parameters
        ----------
        point_a : array-like, shape=[..., 1]
            Point.
        point_b : array-like, shape=[..., 1]
            Point.
        atol : float
            Tolerance.

        Returns
        -------
        dist : array-like, shape=[...]
            Riemannian distance.
        """
        return self._tau(point_a, point_b, atol=atol)

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        Denoting `base_point` by :math:`x` and `end_point` by :math:`y`,

        .. math::

            P_{x \rightarrow y}(v) = \frac{1 - |y|^2}{1 - |x|^2}
            \frac{1 - y \bar{x}}{1 - \bar{y} x} v

        i.e. the ratio of conformal factors times the gyration of the
        Mobius addition, which is a rotation in the complex plane.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., 1]
            Point in the complex Poincare disk.
        direction : array-like, shape=[..., 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
        end_point : array-like, shape=[..., 1]
            Point in the complex Poincare disk to transport to. Unused if
            `direction` is given.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., 1]
            Transported tangent vector at `exp_(base_point)(direction)`.
        """
        if direction is not None:
            end_point = self.exp(direction, base_point)
        elif end_point is None:
            raise ValueError(
                "Either an end_point or a direction must be given to define the"
                " geodesic along which to transport."
            )

        conformal_ratio = (1 - gs.abs(end_point) ** 2) / (1 - gs.abs(base_point) ** 2)
        gyration = (1 - end_point * gs.conj(base_point)) / (
            1 - gs.conj(end_point) * base_point
        )
        return conformal_ratio * gyration * tangent_vec

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        In the complex Poincare disk, it is infinite everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)
