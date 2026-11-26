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
from geomstats import algebra_utils as utils
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.hermitian import Hermitian
from geomstats.geometry.homogeneous_metric import RiemannianSymmetricMetric
from geomstats.geometry.isometry import PointedIsometry
from geomstats.geometry.manifold import _get_default_dtype


class MobiusTransvection(PointedIsometry):
    r"""Transvection of the complex Poincaré disk sending an anchor to zero.

    Denoting the anchor by :math:`a`, the transvection is the Möbius map

    .. math::

        \phi_a(z) = \frac{z - a}{1 - \bar{a} z}

    along the geodesic from :math:`a` to zero. Its inverse is
    :math:`\phi_{-a}`. It is holomorphic, so its differential is the
    multiplication by its complex derivative.

    Parameters
    ----------
    anchor : array-like, shape=[..., 1]
        Point on the complex Poincaré disk sent to zero.
    """

    def __init__(self, anchor):
        super().__init__(anchor, gs.zeros(anchor.shape[-1:], dtype=anchor.dtype))
        self._anchor_conj = gs.conj(anchor)
        self._conformal_factor = 1 - gs.abs(anchor) ** 2

    def _map(self, point, sign):
        """Apply the transvection of anchor `sign * anchor`."""
        return (point - sign * self.anchor) / (1 - sign * self._anchor_conj * point)

    def _tangent(self, tangent_vec, point, sign):
        r"""Apply the differential at `point` of the transvection of `sign * anchor`.

        .. math::

            d\phi_a(z)[v] = \frac{1 - |a|^2}{(1 - \bar{a} z)^2} v
        """
        denominator = 1 - sign * self._anchor_conj * point
        return self._conformal_factor / denominator**2 * tangent_vec

    def __call__(self, base_point):
        """Send a point by the transvection.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point on the complex Poincaré disk.

        Returns
        -------
        image_point : array-like, shape=[..., 1]
            Image of the point.
        """
        return self._map(base_point, 1.0)

    def inverse(self, image_point):
        """Send a point by the inverse transvection.

        Parameters
        ----------
        image_point : array-like, shape=[..., 1]
            Point on the complex Poincaré disk.

        Returns
        -------
        base_point : array-like, shape=[..., 1]
            Preimage of the point.
        """
        return self._map(image_point, -1.0)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Send a tangent vector by the differential of the transvection.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.
        image_point : array-like, shape=[..., 1]
            Image of the base point. Used if `base_point` is not given.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 1]
            Tangent vector at the image of the base point.
        """
        if base_point is None:
            if image_point is None:
                raise ValueError("Either a base point or an image point is needed.")
            base_point = self.inverse(image_point)
        return self._tangent(tangent_vec, base_point, 1.0)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Send a tangent vector by the differential of the inverse transvection.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 1]
            Tangent vector at image point.
        image_point : array-like, shape=[..., 1]
            Image point.
        base_point : array-like, shape=[..., 1]
            Preimage of the image point. Used if `image_point` is not given.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at the preimage of the image point.
        """
        if image_point is None:
            if base_point is None:
                raise ValueError("Either a base point or an image point is needed.")
            image_point = self(base_point)
        return self._tangent(image_tangent_vec, image_point, -1.0)

    def tangent_at_anchor(self, tangent_vec):
        r"""Send a tangent vector at the anchor by the differential.

        At the anchor, the differential reduces to
        :math:`d\phi_a(a)[v] = v / (1 - |a|^2)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at the anchor.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 1]
            Tangent vector at zero.
        """
        return tangent_vec / self._conformal_factor

    def inverse_tangent_at_reference(self, image_tangent_vec):
        r"""Send a tangent vector at zero by the differential of the inverse.

        At zero, the differential of the inverse reduces to
        :math:`d\phi_{-a}(0)[v] = (1 - |a|^2) v`.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 1]
            Tangent vector at zero.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at the anchor.
        """
        return self._conformal_factor * image_tangent_vec


class ComplexPoincareDisk(VectorSpaceOpenSet):
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
        modulus = bound * gs.random.rand(*size)
        angle = 2 * gs.pi * gs.random.rand(*size)
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


class ComplexPoincareDiskMetric(RiemannianSymmetricMetric):
    """Class for the complex Poincaré metric.

    The geometry is only written at zero and is moved to any other base point
    by the Möbius transvection sending that base point to zero. The parallel
    transport has a closed form.
    """

    @property
    def reference_point(self):
        """Reference point, zero.

        Returns
        -------
        reference_point : array-like, shape=[1]
            Zero.
        """
        return gs.zeros(
            self._space.shape, dtype=_get_default_dtype(self._space.complex_repr)
        )

    def isometry_to_reference(self, base_point):
        """Get the isometry sending a base point to zero.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        isometry : MobiusTransvection
            Transvection sending the base point to zero.
        """
        return MobiusTransvection(base_point)

    def inner_product_at_reference(self, tangent_vec_a, tangent_vec_b):
        """Compute the inner product at zero.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., 1]
            Tangent vector at zero.
        tangent_vec_b : array-like, shape=[..., 1]
            Tangent vector at zero.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner product.
        """
        return gs.real(gs.vecdot(tangent_vec_a, tangent_vec_b))

    def exp_at_reference(self, tangent_vec):
        r"""Compute the exponential map at zero.

        .. math::

            \exp_0(v) = \frac{\tanh(|v|)}{|v|} v

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at zero.

        Returns
        -------
        exp : array-like, shape=[..., 1]
            Point on the complex Poincaré disk.
        """
        inv_coef = utils.taylor_exp_even_func(
            gs.abs(tangent_vec) ** 2, utils.inv_tanh_close_0
        )
        return tangent_vec / inv_coef

    def log_at_reference(self, point):
        r"""Compute the logarithm map at zero.

        .. math::

            \log_0(z) = \frac{\operatorname{artanh}(|z|)}{|z|} z

        The modulus of the point is clipped below one.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point on the complex Poincaré disk.

        Returns
        -------
        log : array-like, shape=[..., 1]
            Tangent vector at zero.
        """
        modulus = gs.abs(point)
        point = (1 - gs.atol) / gs.maximum(modulus, 1 - gs.atol) * point
        coef = utils.taylor_exp_even_func(
            gs.abs(point) ** 2, utils.arctanh_card_close_0
        )
        return coef * point

    def squared_dist_at_reference(self, point):
        r"""Compute the squared distance from zero.

        .. math::

            d(0, z)^2 = \operatorname{artanh}(|z|)^2

        The modulus of the point is clipped below one.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point on the complex Poincaré disk.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
            Squared distance between zero and the point.
        """
        modulus = gs.minimum(gs.abs(point), 1 - gs.atol)
        return gs.sum(gs.arctanh(modulus) ** 2, axis=-1)

    def sectional_curvature_at_reference(self, tangent_vec_a, tangent_vec_b):
        """Compute the sectional curvature at zero.

        The complex Poincaré disk has constant curvature -4.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., 1]
            Tangent vector at zero.
        tangent_vec_b : array-like, shape=[..., 1]
            Tangent vector at zero.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at zero.
        """
        return -4.0 * gs.ones_like(
            self.inner_product_at_reference(tangent_vec_a, tangent_vec_b)
        )

    def injectivity_radius_at_reference(self):
        """Compute the injectivity radius at zero.

        The complex Poincaré disk is a Hadamard manifold, so the radius is
        infinite.

        Returns
        -------
        radius : array-like, shape=[]
            Injectivity radius.
        """
        return gs.array(math.inf)

    def metric_matrix(self, base_point):
        """Compute the metric matrix at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        inner_prod_mat : array-like, shape=[..., 1, 1]
            Inner product matrix.
        """
        inner_prod_mat = 1 / (1 - gs.abs(base_point) ** 2) ** 2
        return gs.cast(gs.expand_dims(inner_prod_mat, axis=-1), base_point.dtype)

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
