"""Parameterized curves on any given manifold.

Lead author: Alice Le Brigant.
"""

import copy
import logging
import math

from scipy.interpolate import CubicSpline

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.diffeo import AutodiffDiffeo, Diffeo
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.fiber_bundle import FiberBundle
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.landmarks import Landmarks
from geomstats.geometry.nfold_manifold import NFoldManifold, NFoldMetric
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.vectorization import check_is_batch, get_batch_shape


def forward_difference(array, delta=None, axis=-1):
    """Forward difference in a Euclidean space.

    Points live in R^n, but are a 1d embedding (e.g. a curve).

    Parameters
    ----------
    array : array-like
        Values of a function.
    delta : float
        Spacing between points.
    axis : int
        Axis in which perform the difference.
        Must be given backwards.

    Returns
    -------
    forward_diff : array-like
        Shape in the specified axis reduces by one.
    """
    n = array.shape[axis]
    if delta is None:
        delta = 1 / (n - 1)

    none_slc = (slice(None),) * (abs(axis) - 1)

    slc = (..., slice(1, n)) + none_slc
    forward = array[slc]

    slc = (..., slice(0, n - 1)) + none_slc
    center = array[slc]
    return (forward - center) / delta


def centered_difference(array, delta=None, axis=-1, endpoints=False):
    """Centered difference in a Euclidean space.

    Points live in R^n, but are a 1d embedding (e.g. a curve).

    Parameters
    ----------
    array : array-like
        Values of a function.
    delta : float
        Spacing between points.
    axis : int
        Axis in which perform the difference.
        Must be given backwards.
    endpoints : bool
        If True, endpoints are computed by backward and forward differences,
        respectively.

    Returns
    -------
    centered_diff : array-like
        Same shape as array.
    """
    n = array.shape[axis]
    if delta is None:
        delta = 1 / (n - 1)

    none_slc = (slice(None),) * (abs(axis) - 1)

    slc = (..., slice(2, n)) + none_slc
    forward = array[slc]

    slc = (..., slice(0, n - 2)) + none_slc
    backward = array[slc]
    diff = (forward - backward) / (2 * delta)

    if endpoints:
        slc_left = (..., [0]) + none_slc
        slc_left_forward = (..., [1]) + none_slc
        diff_left = (array[slc_left_forward] - array[slc_left]) / delta

        slc_right = (..., [-1]) + none_slc
        slc_right_backward = (..., [-2]) + none_slc
        diff_right = (array[slc_right] - array[slc_right_backward]) / delta

        slc_right = (..., [-1]) + none_slc
        return gs.concatenate((diff_left, diff, diff_right), axis=axis)

    return diff


def second_centered_difference(array, delta=None, axis=-1):
    """Second centered difference in a Euclidean space.

    Points live in R^n, but are a 1d embedding (e.g. a curve).

    Parameters
    ----------
    array : array-like
        Values of a function.
    delta : float
        Spacing between points.
    axis : int
        Axis in which perform the difference.
        Must be given backwards.

    Returns
    -------
    second_centered_diff : array-like
        Shape in the specified axis reduces by two (endpoints).
    """
    n = array.shape[axis]
    if delta is None:
        delta = 1 / (n - 1)

    none_slc = (slice(None),) * (abs(axis) - 1)

    slc = (..., slice(2, n)) + none_slc
    forward = array[slc]

    slc = (..., slice(0, n - 2)) + none_slc
    backward = array[slc]

    slc = (..., slice(1, n - 1)) + none_slc
    central = array[slc]

    return (forward + backward - 2 * central) / (delta**2)


def insert_zeros(array, axis=-1, end=False):
    """Insert zeros in a given array.

    Insert zeros while taking care of

    Parameters
    ----------
    array : array-like
    axis : int
        Axis in which insert the zeros.
        Must be given backwards.
    end : bool
        If True, zeros are introduced at the end.

    Returns
    -------
    array_with_zeros : array-like
        Shape in the specified axis increases by one.
    """
    array_ndim = len(array.shape[axis:])

    batch_shape = get_batch_shape(array_ndim, array)

    shape = batch_shape + (1,) + array.shape[len(batch_shape) + 1 :]
    zeros = gs.zeros(shape)

    first, second = (array, zeros) if end else (zeros, array)
    return gs.concatenate((first, second), axis=-(array_ndim))


class DiscreteCurvesStartingAtOrigin(NFoldManifold):
    r"""Space of discrete curves modulo translations.

    Each individual curve is represented by a 2d-array of shape `[
    k_sampling_points - 1, ambient_dim]`.

    This space corresponds to the space of immersions defined below, i.e. the
    space of smooth functions from an interval I into the ambient Euclidean
    space M, with non-vanishing derivative.

    .. math::
        Imm(I, M)=\{c \in C^{\infty}(I, M) \|c'(t)\|\neq 0 \forall t \in I \},

    where the interval of parameters I is taken to be I = [0, 1]
    without loss of generality.

    Parameters
    ----------
    ambient_dim : int
        Dimension of the ambient Euclidean space in which curves take values.
    k_sampling_points : int
        Number of sampling points.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, ambient_dim=2, k_sampling_points=10, equip=True):
        ambient_manifold = Euclidean(ambient_dim)
        super().__init__(ambient_manifold, k_sampling_points - 1, equip=equip)

        self._quotient_map = {
            (SRVTranslationMetric, "reparametrizations"): (
                SRVTranslationReparametrizationBundle,
                SRVTranslationReparametrizationQuotientMetric,
            ),
        }
        self._sphere = Hypersphere(dim=ambient_dim - 1)

    def new(self, equip=True):
        """Create manifold with same parameters."""
        return DiscreteCurvesStartingAtOrigin(
            ambient_dim=self.ambient_manifold.dim,
            k_sampling_points=self.k_sampling_points,
            equip=equip,
        )

    @property
    def ambient_manifold(self):
        """Manifold in which curves take values."""
        return self.base_manifold

    @property
    def k_sampling_points(self):
        """Number of sampling points for the discrete curves."""
        return self.n_copies + 1

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return SRVTranslationMetric

    def insert_origin(self, point):
        """Insert origin as first element of point."""
        return insert_zeros(point, axis=-self.point_ndim)

    def projection(self, point):
        """Project a point from discrete curves.

        Removes translation and origin.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]

        Returns
        -------
        proj_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
        """
        if point.shape[-2] == self.k_sampling_points - 1:
            return gs.copy(point)

        return (point[..., :, :] - gs.expand_dims(point[..., 0, :], axis=-2))[
            ..., 1:, :
        ]

    def random_point(self, n_samples=1, bound=1.0):
        """Sample random curves.

        Sampling on the sphere to avoid chaotic curves.
        """
        sampling_times = gs.linspace(0.0, 1.0, self.k_sampling_points)

        initial_point = self._sphere.random_point(n_samples)
        initial_tangent_vec = self._sphere.random_tangent_vec(initial_point)

        point = self._sphere.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )(sampling_times)
        return self.projection(point)


class SRVTransform(Diffeo):
    """SRV transform.

    Diffeomorphism between discrete curves starting at origin with
    `k_sampling_points` and landmarks with `k_sampling_points - 1`.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.
    k_sampling_points : int
        Number of sampling points.

    Notes
    -----
    It is currently only implemented for the Euclidean ambient manifold.
    """

    def __init__(self, ambient_manifold, k_sampling_points):
        self.ambient_manifold = ambient_manifold
        self.k_sampling_points = k_sampling_points

        self._point_ndim = self.ambient_manifold.point_ndim + 1

    def diffeomorphism(self, base_point):
        r"""Square Root Velocity Transform (SRVT).

        Compute the square root velocity representation of a curve. The
        velocity is computed using the log map.

        In the case of several curves, an index selection procedure allows to
        get rid of the log between the end point of curve[k, :, :] and the starting
        point of curve[k + 1, :, :].

        .. math::
            Q(c) = c'/ |c'|^{1/2}

        Parameters
        ----------
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.

        Returns
        -------
        image_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            SRV representation.
        """
        ndim = self._point_ndim
        base_point_with_origin = insert_zeros(base_point, axis=-ndim)

        velocity = forward_difference(base_point_with_origin, axis=-ndim)

        pointwise_velocity_norm = self.ambient_manifold.metric.norm(
            velocity, base_point_with_origin[..., :-1, :]
        )
        return gs.einsum(
            "...ij,...i->...ij", velocity, 1.0 / gs.sqrt(pointwise_velocity_norm)
        )

    def inverse_diffeomorphism(self, image_point):
        r"""Inverse of the Square Root Velocity Transform (SRVT).

        Retrieve a curve from its square root velocity representation.

        .. math::
            c(t) = c(0) + \int_0^t q(s) |q(s)|ds

        with:
        - c the curve that can be retrieved only up to a translation,
        - q the srv representation of the curve,
        - c(0) the starting point of the curve.


        See [Sea2011]_ Section 2.1 for details.

        It performs numerical integration on a manifold.

        Parameters
        ----------
        image_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            SRV representation.

        Returns
        -------
        curve : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.
        """
        image_point_norm = self.ambient_manifold.metric.norm(image_point)

        dt = 1 / (self.k_sampling_points - 1)

        pointwise_delta_points = gs.einsum(
            "...,...i->...i", dt * image_point_norm, image_point
        )

        return gs.cumsum(pointwise_delta_points, axis=-2)

    def tangent_diffeomorphism(self, tangent_vec, base_point=None, image_point=None):
        r"""Differential of the square root velocity transform.

        ..math::
            (h, c) -> dQ_c(h) = |c'|^(-1/2} * (h' - 1/2 * <h',v>v)
            v = c'/|c'|

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Tangent vector to curve.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambiend_dim]
            Discrete curve.
        image_point : array-like, shape=[..., k_sampling_points - 1, ambiend_dim]
            SRV representation.

        Returns
        -------
        d_srv_vec : array-like, shape=[..., k_sampling_points - 1, ambient_dim,]
            Differential of the square root velocity transform at curve
            evaluated at tangent_vec.
        """
        if base_point is None:
            base_point = self.inverse_diffeomorphism(image_point)

        ndim = self._point_ndim
        base_point_with_origin = insert_zeros(base_point, axis=-ndim)
        tangent_vec_with_zeros = insert_zeros(tangent_vec, axis=-ndim)

        d_vec = forward_difference(tangent_vec_with_zeros, axis=-ndim)
        velocity_vec = forward_difference(base_point_with_origin, axis=-ndim)

        velocity_norm = self.ambient_manifold.metric.norm(velocity_vec)
        unit_velocity_vec = gs.einsum(
            "...ij,...i->...ij", velocity_vec, 1 / velocity_norm
        )

        pointwise_inner_prod = self.ambient_manifold.metric.inner_product(
            d_vec, unit_velocity_vec, base_point_with_origin[..., :-1, :]
        )
        d_vec_tangential = gs.einsum(
            "...ij,...i->...ij",
            unit_velocity_vec,
            pointwise_inner_prod,
        )
        d_srv_vec = d_vec - 1 / 2 * d_vec_tangential
        d_srv_vec = gs.einsum(
            "...ij,...i->...ij", d_srv_vec, 1 / velocity_norm ** (1 / 2)
        )

        return d_srv_vec

    def inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point=None, base_point=None
    ):
        r"""Inverse of differential of the square root velocity transform.

        .. math::
            (c, k) -> h, where dQ_c(h)=k and h' = |c'| * (k + <k,v> v)

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Tangent vector to SRV representation.
        image_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            SRV representation.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.

        Returns
        -------
        tangent_vec : array-like, shape=[..., ambient_dim]
            Inverse of the differential of the square root velocity transform at
            curve evaluated at tangent_vec.
        """
        if base_point is None:
            base_point = self.inverse_diffeomorphism(image_point)

        ndim = self._point_ndim
        base_point_with_origin = insert_zeros(base_point, axis=-ndim)

        position = base_point_with_origin[..., :-1, :]
        velocity_vec = forward_difference(base_point_with_origin, axis=-ndim)
        velocity_norm = self.ambient_manifold.metric.norm(velocity_vec, position)
        unit_velocity_vec = gs.einsum(
            "...ij,...i->...ij", velocity_vec, 1 / velocity_norm
        )

        pointwise_inner_prod = self.ambient_manifold.metric.inner_product(
            image_tangent_vec,
            unit_velocity_vec,
            position,
        )
        tangent_vec_tangential = gs.einsum(
            "...ij,...i->...ij", unit_velocity_vec, pointwise_inner_prod
        )
        d_vec = image_tangent_vec + tangent_vec_tangential
        d_vec = gs.einsum("...ij,...i->...ij", d_vec, velocity_norm ** (1 / 2))
        increment = d_vec / (self.k_sampling_points - 1)

        return gs.cumsum(increment, axis=-2)


class FTransform(AutodiffDiffeo):
    r"""FTransform.

    The f_transform is defined on the space of curves
    quotiented by translations, which is identified with the space
    of curves with their first sampling point located at 0:

    .. math::
        curve(0) = (0, 0)

    The f_transform is given by the formula:

    .. math::
        Imm(I, R^2) / R^2 \mapsto C^\infty(I, R^2\backslash\{0\})
        c \mapsto 2b |c'|^{1/2} (\frac{c'}{|c'|})^{a/(2b)}

    where the identification :math:`C = R^2` is used and
    the exponentiation is a complex exponentiation, which can make
    the f_transform not well-defined:

    .. math::
        f(c) = 2b r^{1/2}\exp(i\theta * a/(2b)) * \exp(ik\pi * a/b)

    where (r, theta) is the polar representation of c', and for
    any :math:`k \in Z`.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.
    k_sampling_points : int
        Number of sampling points.
    a : float
        Bending parameter.
    b : float
        Stretching parameter.

    Notes
    -----
    It is currently only implemented for the Euclidean ambient manifold with
    dimension 2.

    f_transform is a bijection if and only if a/2b=1.

    If a/2b is an integer not equal to 1:
    - then f_transform is well-defined but many-to-one.

    If a/2b is not an integer:
    - then f_transform is multivalued,
    - and f_transform takes finitely many values if and only if a 2b is rational.
    """

    def __init__(self, ambient_manifold, k_sampling_points, a=1.0, b=None):
        self._check_ambient_manifold(ambient_manifold)

        self.a = a
        if b is None:
            b = a / 2
        self.b = b
        self.ambient_manifold = ambient_manifold
        self.k_sampling_points = k_sampling_points

        shape = (k_sampling_points - 1,) + self.ambient_manifold.shape
        super().__init__(shape, shape)

    def _check_ambient_manifold(self, ambient_manifold):
        if not (isinstance(ambient_manifold, Euclidean) and ambient_manifold.dim == 2):
            raise NotImplementedError(
                "This transformation is only implemented for planar curves:\n"
                "ambient_manifold must be a plane, but it is:\n"
                f"{ambient_manifold} of dimension {ambient_manifold.dim}."
            )

    def _cartesian_to_polar(self, tangent_vec):
        """Compute polar coordinates of a tangent vector from the cartesian ones.

        This function is an auxiliary function used for the computation
        of the f_transform and its inverse, and is applied to the derivative
        of a curve.

        See [KN2018]_ for details.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k, ambient_dim]
            Tangent vector, representing the derivative c' of a discrete curve c.

        Returns
        -------
        polar_tangent_vec : array-like, shape=[..., k, ambient_dim]
        """
        k_sampling_points = tangent_vec.shape[-2]
        norms = self.ambient_manifold.metric.norm(tangent_vec)
        arg_0 = gs.arctan2(tangent_vec[..., 0, 1], tangent_vec[..., 0, 0])
        args = [arg_0]

        for i in range(1, k_sampling_points):
            point, last_point = tangent_vec[..., i, :], tangent_vec[..., i - 1, :]
            prod = self.ambient_manifold.metric.inner_product(point, last_point)
            cosine = prod / (norms[..., i] * norms[..., i - 1])
            angle = gs.arccos(gs.clip(cosine, -1, 1))
            det = gs.linalg.det(gs.stack([last_point, point], axis=-1))
            orientation = gs.sign(det)
            arg = args[-1] + orientation * angle
            args.append(arg)

        args = gs.stack(args, axis=-1)
        return gs.stack([norms, args], axis=-1)

    def _polar_to_cartesian(self, polar_tangent_vec):
        """Compute the cartesian coordinates of a tangent vector from polar ones.

        This function is an auxiliary function used for the computation
        of the f_transform.

        Parameters
        ----------
        polar_tangent_vec : array-like, shape=[..., k, ambient_dim]

        Returns
        -------
        tangent_vec : array-like, shape=[..., k, ambient_dim]
            Tangent vector.
        """
        tangent_vec_x = gs.cos(polar_tangent_vec[..., :, 1])
        tangent_vec_y = gs.sin(polar_tangent_vec[..., :, 1])
        norms = polar_tangent_vec[..., :, 0]
        unit_tangent_vec = gs.stack((tangent_vec_x, tangent_vec_y), axis=-1)

        return norms[..., :, None] * unit_tangent_vec

    def diffeomorphism(self, base_point):
        r"""Compute the f_transform of a curve.

        The implementation uses formula (3) from [KN2018]_ , i.e. choses
        the representative corresponding to k = 0.

        Parameters
        ----------
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.

        Returns
        -------
        image_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            F_transform of the curve.
        """
        coeff = self.k_sampling_points - 1

        base_point_with_origin = insert_zeros(base_point, axis=-self._space_point_ndim)

        velocity = coeff * (
            base_point_with_origin[..., 1:, :] - base_point_with_origin[..., :-1, :]
        )
        polar_velocity = self._cartesian_to_polar(velocity)
        speeds = polar_velocity[..., :, 0]
        args = polar_velocity[..., :, 1]

        f_args = args * (self.a / (2 * self.b))
        f_norms = 2 * self.b * gs.sqrt(speeds)
        f_polar = gs.stack([f_norms, f_args], axis=-1)

        return self._polar_to_cartesian(f_polar)

    def inverse_diffeomorphism(self, image_point):
        r"""Compute the inverse F_transform of a transformed curve.

        This only works if a / (2b) <= 1.
        See [KN2018]_ for details.

        When the f_transform is many-to-one, one antecedent is chosen.

        Parameters
        ----------
        image_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            F tranform representation of a discrete curve.

        Returns
        -------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Curve starting at the origin retrieved from its square-root velocity.
        """
        coef = self.k_sampling_points - 1

        f_polar = self._cartesian_to_polar(image_point)
        f_norms = f_polar[..., :, 0]
        f_args = f_polar[..., :, 1]

        dt = 1 / coef

        delta_points_x = gs.einsum(
            "...i,...i->...i", dt * f_norms**2, gs.cos(2 * self.b / self.a * f_args)
        )
        delta_points_y = gs.einsum(
            "...i,...i->...i", dt * f_norms**2, gs.sin(2 * self.b / self.a * f_args)
        )

        delta_points = gs.stack((delta_points_x, delta_points_y), axis=-1)

        delta_points = 1 / (4 * self.b**2) * delta_points

        return gs.cumsum(delta_points, axis=-2)


class L2CurvesMetric(NFoldMetric):
    """L2 metric on the space of discrete curves.

    L2 metric on the space of regularly sampled discrete curves
    defined on the unit interval. The inner product between two tangent vectors
    is given by the integral of the ambient space inner product, approximated by
    a left Riemann sum.
    """

    def __init__(self, space):
        super().__init__(
            space=space,
            signature=(math.inf, 0, 0),
        )

    @staticmethod
    def riemann_sum(func):
        """Compute the left Riemann sum approximation of the integral.

        Compute the left Riemann sum approximation of the integral of a
        function func defined on the unit interval, given by sample points
        at regularly spaced times
        ..math::
            t_i = i / (k_landmarks),
            i = 0, ..., k_landmarks - 1
        (last time is missing).

        Parameters
        ----------
        func : array-like, shape=[..., k_landmarks]
            Sample points of a function at regularly spaced times.

        Returns
        -------
        riemann_sum : array-like, shape=[..., ]
            Left Riemann sum.
        """
        k_sampling_points_minus_one = func.shape[-1]
        dt = 1 / k_sampling_points_minus_one
        return dt * gs.sum(func, axis=-1)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute L2 inner product between two tangent vectors.

        The inner product is the integral of the ambient space inner product,
        approximated by a left Riemann sum.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector to a curve, i.e. infinitesimal vector field
            along a curve.
        tangent_vec_b : array-like, shape=[..., k_landmarks, ambient_dim]
            Tangent vector to a curve, i.e. infinitesimal vector field
            along a curve.
        base_point : array-like, shape=[..., k_landmarks, ambient_dim]
            Discrete curve defined on the unit interval [0, 1].

        Return
        ------
        inner_prod : array_like, shape=[...]
            L2 inner product between tangent_vec_a and tangent_vec_b.
        """
        inner_products = self.pointwise_inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        return self.riemann_sum(inner_products)


class ElasticTranslationMetric(PullbackDiffeoMetric):
    """Elastic metric on the space of discrete curves.

    Family of elastic metric parametrized by bending and stretching parameters
    a and b. These can be obtained as pullbacks of the L2 metric by the F-transforms.

    See [NK2018]_ for details.

    Parameters
    ----------
    a : float
        Bending parameter.
    b : float
        Stretching parameter.

    References
    ----------
    .. [KN2018] S. Kurtek and T. Needham,
        "Simplifying transforms for general elastic metrics on the space of
        plane curves", arXiv:1803.10894 [math.DG], 29 Mar 2018.
    """

    def __init__(
        self,
        space,
        a,
        b=None,
    ):
        image_space = self._instantiate_image_space(space)
        diffeo = FTransform(space.ambient_manifold, space.k_sampling_points, a, b)

        super().__init__(
            space=space,
            diffeo=diffeo,
            image_space=image_space,
            signature=(math.inf, 0, 0),
        )

    def _instantiate_image_space(self, space):
        image_space = Landmarks(
            ambient_manifold=space.ambient_manifold,
            k_landmarks=space.k_sampling_points - 1,
            equip=False,
        )
        image_space.equip_with_metric(L2CurvesMetric)
        return image_space


class SRVTranslationMetric(PullbackDiffeoMetric):
    """Square Root Velocity metric on the space of discrete curves.

    The SRV metric is equivalent to the elastic metric chosen with
    - bending parameter a = 1,
    - stretching parameter b = 1/2.
    It can be obtained as the pullback of the L2 metric by the Square Root
    Velocity Function.

    See [Sea2011]_ for details.

    References
    ----------
    .. [Sea2011] A. Srivastava, E. Klassen, S. H. Joshi and I. H. Jermyn,
        "Shape Analysis of Elastic Curves in Euclidean Spaces,"
        in IEEE Transactions on Pattern Analysis and Machine Intelligence,
        vol. 33, no. 7, pp. 1415-1428, July 2011.
    """

    def __init__(self, space):
        self._check_ambient_manifold(space.ambient_manifold)

        image_space = self._instantiate_image_space(space)

        diffeo = SRVTransform(space.ambient_manifold, space.k_sampling_points)
        super().__init__(space=space, diffeo=diffeo, image_space=image_space)

    def _check_ambient_manifold(self, ambient_manifold):
        if not isinstance(ambient_manifold, Euclidean):
            raise AssertionError(
                "This metric is only "
                "implemented for discrete curves embedded "
                "in a Euclidean space."
            )

    def _instantiate_image_space(self, space):
        image_space = Landmarks(
            ambient_manifold=space.ambient_manifold,
            k_landmarks=space.k_sampling_points - 1,
            equip=False,
        )
        image_space.equip_with_metric(L2CurvesMetric)
        return image_space


class IterativeHorizontalGeodesicAligner:
    """Iterative horizontal geodesic algorithm.

    The horizontal geodesic is computed by an interative procedure where
    the initial curve stays fixed and the sampling points are moved on the
    end curve to obtain its optimal parametrization with respect to the
    initial curve. This optimal matching algorithm sets current_end_curve
    to be the end curve and iterates three steps:
    1) compute the geodesic between the initial curve and current_end_curve
    2) compute the path of reparametrizations that transforms this geodesic
    into a horizontal path of curves
    3) invert this path of reparametrizations to find the horizontal path
    and update current_end_curve to be its end point.
    The algorithm stops when the new current_end_curve is sufficiently
    close to the former current_end_curve.

    Parameters
    ----------
    n_time_grid : int
        Number of times in which compute the geodesic.
    threshold : float
        When the difference between the new end curve and the current end
        curve becomes lower than this threshold, the algorithm stops.
        Optional, default: 1e-3.
    max_iter : int
        Maximum number of iterations.
    save_history : bool
        If True, history is saved in a `self.history`.
    """

    def __init__(
        self, n_time_grid=100, threshold=1e-3, max_iter=20, save_history=False
    ):
        self.n_time_grid = n_time_grid
        self.threshold = threshold
        self.max_iter = max_iter
        self.save_history = save_history
        if save_history:
            self.history = None

    def _construct_reparametrization(self, vertical_norm, space_deriv_norm):
        r"""Construct path of reparametrizations.

        Construct path of reparametrizations phi(t, u) that transforms
        a path of curves c(t, u) into a horizontal path of curves, i.e.
        :math:`d/dt c(t, phi(t, u))` is a horizontal vector.

        Parameters
        ----------
        vertical_norm : array-like, shape=[n_times - 1, k_sampling_points]
            Pointwise norm of the vertical part of the time derivative of
            the path of curves.
        space_deriv_norm : array-like, shape=[n_times - 1, k_sampling_points]
            Pointwise norm of the space derivative of the path of curves.

        Returns
        -------
        repar : array-like, shape=[n_times, k_sampling_points]
            Path of parametrizations, such that the path of curves
            composed with the path of parametrizations is a horizontal
            path.
        """
        n_times = vertical_norm.shape[0] + 1
        k_sampling_points = vertical_norm.shape[1]

        quotient = vertical_norm / space_deriv_norm

        repar = gs.linspace(0.0, 1.0, k_sampling_points)
        repars = [repar]
        for i in range(n_times - 1):
            repar_diff = forward_difference(repar, axis=-1)

            repar_space_deriv = gs.where(
                vertical_norm[i, 1:-1] > 0,
                repar_diff[1:],
                repar_diff[:-1],
            )
            repar_time_deriv = repar_space_deriv * quotient[i, 1:-1]
            repar_i_mid = repar[1:-1] + repar_time_deriv / n_times

            repar = repar_i = gs.concatenate(
                [gs.array([0.0]), repar_i_mid, gs.array([1.0])]
            )
            repars.append(repar_i)

            test_repar = repar[1:] - repar[:-1] < 0
            if gs.any(test_repar):
                raise ValueError(
                    "Warning: phi(t) is non increasing for at least "
                    "one time t. Solution may be inaccurate."
                )

        return gs.stack(repars)

    def _invert_reparametrize_single(self, t_space, repar, point):
        spline = CubicSpline(t_space, point, axis=0)
        repar_inverse = CubicSpline(repar, t_space)
        return gs.from_numpy(spline(repar_inverse(t_space)))

    def _invert_reparametrization(
        self,
        t_space,
        repar,
        path_of_curves,
        repar_inverse_end,
        spline_end,
    ):
        r"""Invert path of reparametrizations.

        Given a path of curves c(t, u) and a path of reparametrizations
        phi(t, u), compute:
        :math:`c(t, phi_inv(t, u))` where `phi_inv(t, .) = phi(t, .)^{-1}`

        The computation for the last time t=1 is done differently, using
        the spline function associated to the end curve and the composition
        of the inverse reparametrizations contained in rep_inverse_end:
        :math:`spline_end_curve ° phi_inv(1, .) ° ... ° phi_inv(0, .)`.

        Parameters
        ----------
        repar : array-like, shape=[n_times, k_sampling_points]
            Path of reparametrizations.
        path_of_curves : array-like, shape=[n_times, k_sampling_points, ambient_dim]
            Path of curves.
        repar_inverse_end: list
            List of the inverses of the reparametrizations applied to
            the end curve during the optimal matching algorithm.

        Returns
        -------
        reparametrized_path : array-like,
            shape=[n_times, k_sampling_points, ambient_dim]
            Path of curves composed with the inverse of the path of
            reparametrizations.
        """
        initial_curve = path_of_curves[0]
        reparametrized_path = [initial_curve] + [
            self._invert_reparametrize_single(
                t_space,
                repar_i,
                point,
            )
            for repar_i, point in zip(repar[1:-1], path_of_curves[1:-1])
        ]

        repar_inverse_end.append(CubicSpline(repar[-1], t_space))
        arg = t_space
        for repar_inverse in reversed(repar_inverse_end):
            arg = repar_inverse(arg)
        end_curve_repar = gs.from_numpy(spline_end(arg))

        reparametrized_path.append(end_curve_repar)
        return gs.stack(reparametrized_path)

    def _iterate(
        self,
        bundle,
        times,
        t_space,
        initial_point,
        end_point,
        repar_inverse_end,
        spline_end,
    ):
        """Perform one step of the aligment algorithm."""
        ndim = bundle.total_space.point_ndim

        total_space_geod_fun = bundle.total_space.metric.geodesic(
            initial_point=initial_point, end_point=end_point
        )
        geod_points = total_space_geod_fun(times)
        geod_points_with_origin = insert_zeros(geod_points, axis=-ndim)

        time_deriv = forward_difference(geod_points, axis=-(ndim + 1))
        _, vertical_norm = bundle.vertical_projection(
            time_deriv, geod_points[:-1], return_norm=True
        )
        vertical_norm = insert_zeros(vertical_norm, axis=-1)

        space_deriv = centered_difference(
            geod_points_with_origin, axis=-ndim, endpoints=True
        )[:-1]

        pointwise_space_deriv_norm = bundle.total_space.ambient_manifold.metric.norm(
            space_deriv, geod_points_with_origin[:-1]
        )

        repar = self._construct_reparametrization(
            vertical_norm,
            pointwise_space_deriv_norm,
        )

        horizontal_path = self._invert_reparametrization(
            t_space, repar, geod_points_with_origin, repar_inverse_end, spline_end
        )

        return horizontal_path

    def _discrete_horizontal_geodesic_single(self, bundle, initial_point, end_point):
        """Horizontal geodesic between two curves.

        Parameters
        ----------
        initial_point : array-like, shape=[k_sampling_points, ambient_dim]
            Initial discrete curve.
        end_point : array-like, shape=[k_sampling_points, ambient_dim]
            End discrete curve.

        Returns
        -------
        horizontal_geod_points : array, shape=[n_time_grid, k - 1, ambient_dim]
            Geodesic points.
        """
        times = gs.linspace(0.0, 1.0, self.n_time_grid)

        k_sampling_points = bundle.total_space.k_sampling_points
        t_space = gs.linspace(0.0, 1.0, k_sampling_points)

        end_point_with_origin = insert_zeros(
            end_point, axis=-bundle.total_space.point_ndim
        )
        spline_end = CubicSpline(t_space, end_point_with_origin, axis=0)

        current_end_point = end_point
        repar_inverse_end = []
        for _ in range(self.max_iter):
            horizontal_path_with_origin = self._iterate(
                bundle,
                times,
                t_space,
                initial_point,
                current_end_point,
                repar_inverse_end,
                spline_end,
            )

            new_end_point = horizontal_path_with_origin[-1][1:]
            gap = (
                gs.sum(gs.linalg.norm(new_end_point - current_end_point, axis=-1) ** 2)
            ) ** (1 / 2)
            current_end_point = new_end_point

            if gap < self.threshold:
                break
        else:
            logging.warning(
                "Maximum number of iterations %d reached. The result may be inaccurate",
                self.max_iter,
            )

        if self.save_history:
            self.history = dict(
                spline=spline_end,
                repar_inverse=repar_inverse_end,
            )

        return horizontal_path_with_origin[..., 1:, :]

    def discrete_horizontal_geodesic(self, bundle, initial_point, end_point):
        """Discrete horizontal geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Initial discrete curve.
        end_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            End discrete curve.

        Returns
        -------
        geod_points : array, shape=[..., n_time_grid, k - 1, ambient_dim]
        """
        is_batch = check_is_batch(
            bundle.total_space.point_ndim,
            initial_point,
            end_point,
        )
        if not is_batch:
            return self._discrete_horizontal_geodesic_single(
                bundle, initial_point, end_point
            )

        if initial_point.ndim != end_point.ndim:
            initial_point, end_point = gs.broadcast_arrays(initial_point, end_point)

        return gs.stack(
            [
                self._discrete_horizontal_geodesic_single(
                    bundle, initial_point_, end_point_
                )
                for initial_point_, end_point_ in zip(initial_point, end_point)
            ]
        )

    def align(self, bundle, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Initial discrete curve.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            End discrete curve.

        Returns
        -------
        aligned : array-like, shape=[..., k_sampling_points - 1, ambient_dim
            Action of the optimal g on point.
        """
        return self.discrete_horizontal_geodesic(bundle, base_point, point)[
            ..., -1, :, :
        ]


class DynamicProgrammingAligner:
    r"""Align two curves through dynamic programming.

    Find the reparametrization gamma of end_curve that minimizes the distance
    between initial_curve and end_curve reparametrized by gamma, and output
    the corresponding distance, using a dynamic programming algorithm.

    The objective can be expressed in terms of square root velocity (SRV)
    representations: it is equivalent to finding the gamma that maximizes
    the L2 scalar product between initial_srv and end_srv@gamma where initial_srv
    is the SRV representation of the initial curve and end_srv@gamma is the SRV
    representation of the end curve reparametrized by gamma, i.e
    .. math::
    end_srv@\gamma(t) = end_srv(\gamma(t))\cdot|\gamma(t)|^\frac{1}{2}

    The dynamic programming algorithm assumes that for every subinterval
    :math: '\left[\frac{i}{n},\frac{i+1}{n}\right]' of :math: '\left[0,1\right]',
    gamma is linear.

    Parameters
    ----------
    n_space_grid : int
        Number of subintervals in which the reparametrization is linear.
        Optinal, default: 100.
    max_slope : int
        Maximum slope allowed for a reparametrization.
        Optional, default: 6.

    References
    ----------
    [WAJ2007] M. Washington, S. Anuj & H. Joshi,
    "On Shape of Plane Elastic Curves", in International Journal of Computer
    Vision. 73(3):307-324, 2007.
    """

    def __init__(self, n_space_grid=100, max_slope=6.0):
        self.n_space_grid = n_space_grid
        self.max_slope = max_slope

    def _resample_srv_function(self, srv_function, k_sampling_points):
        """Resample SRV function of a discrete curve.

        Parameters
        ----------
        srv_function : array, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.

        Returns
        -------
        srv : array, shape=[..., n_space_grid, ambient_dim]
            SRV function of the curve at the right size.
        """
        n_space_grid = self.n_space_grid
        i = gs.array(range(n_space_grid))

        ratio = (k_sampling_points - 1) / n_space_grid
        indices = gs.cast(gs.floor(i * ratio), dtype=int)

        return srv_function[..., indices, :]

    @staticmethod
    def _compute_integral_restricted(srv_1, srv_2, x_min, x_max, y_min, y_max):
        r"""Compute the value of an integral over a subinterval.

        Compute n * the value of the integral of
        .. math::
        srv_1(t)\cdotsrv_2(\gamma(t))\cdot|\gamma(t)|^\frac{1}{2}
        over :math: '\left[\x_min,x_max\right]' where gamma restricted to
        :math: '\left[\x_min,x_max\right]' is a linear.

        Parameters
        ----------
        srv_1 : array, shape=[n, ambient_dim]
            SRV function of the initial curve.
        srv_2 : array, shape=[n, ambient_dim]
            SRV function of the end curve.
        x_min : int
            Beginning of the subinterval.
        x_max : int
            End of the subinterval.
        y_min : int
            Value of gamma at x_min.
        y_max : int
            Value of gamma at x_max.

        Returns
        -------
        value : float
            Value of the integral described above.
        """
        gamma_slope = (y_max - y_min) / (x_max - x_min)

        list_l = list(range(x_min, x_max + 1))
        list_k = [(k - y_min) / gamma_slope + x_min for k in range(y_min, y_max + 1)]

        lower_bound = x_min
        i = 1
        j = 1

        value = 0.0
        while i < x_max - x_min + 1 and j < y_max - y_min + 1:
            upper_bound = min(list_l[i], list_k[j])
            length = upper_bound - lower_bound
            value += length * gs.dot(srv_1[x_min + i - 1], srv_2[y_min + j - 1])

            if list_l[i] == list_k[j]:
                i += 1
                j += 1
            elif list_l[i] < list_k[j]:
                i += 1
            else:
                j += 1
            lower_bound = upper_bound

        return math.pow(gamma_slope, 1 / 2) * value

    def _reparametrize(self, curve, gamma):
        """Reparametrize curve by gamma.

        Parameters
        ----------
        curve : array, shape=[k_sampling_points, ambient_dim]
            Discrete curve.
        gamma : array, shape=[n_subinterval]
            Parametrization of a curve.

        Returns
        -------
        new_curve : array , shape=[k_sampling_points, ambient_dim]
            Curve reparametrized by gamma.
        """
        n_space_grid = self.n_space_grid
        k_sampling_points = curve.shape[-2]

        new_curve = gs.zeros(curve.shape, dtype=float)
        n_subinterval = len(gamma)
        list_gamma_slope = gs.zeros(n_space_grid + 1, dtype=float)
        list_gamma_constant = gs.zeros(n_space_grid + 1, dtype=float)

        new_curve[0] = curve[0]
        new_curve[-1] = curve[-1]

        for k in range(1, n_subinterval):
            (i_depart, j_depart) = gamma[k - 1]
            (i_arrive, j_arrive) = gamma[k]
            gamma_slope = (j_arrive - j_depart) / (i_arrive - i_depart)
            gamma_constant = j_depart - i_depart * gamma_slope

            for i in range(i_depart, i_arrive):
                list_gamma_slope[i] = gamma_slope
                list_gamma_constant[i] = gamma_constant

        ratio_k = (n_space_grid - 1) / k_sampling_points
        ratio_n = (k_sampling_points - 1) / n_space_grid
        for k in range(1, k_sampling_points):
            indice_n = int(gs.floor(k * ratio_k))
            gamma_indice_n = (k * ratio_k) * list_gamma_slope[
                indice_n
            ] + list_gamma_constant[indice_n]
            gamma_indice_k = gamma_indice_n * ratio_n
            indice_k = int(gs.floor(gamma_indice_k))
            alpha = gamma_indice_k - indice_k

            new_curve[k] = curve[indice_k] * (1 - alpha) + curve[indice_k + 1] * alpha

        return new_curve

    def _compute_squared_dist(self, initial_srv, end_srv, tableau):
        """Compute squared distance using algorithmic information."""
        n_space_grid = self.n_space_grid

        norm_squared_initial_srv = (
            self._compute_integral_restricted(
                initial_srv, initial_srv, 0, n_space_grid, 0, n_space_grid
            )
            / n_space_grid
        )
        norm_squared_end_srv = (
            self._compute_integral_restricted(
                end_srv, end_srv, 0, n_space_grid, 0, n_space_grid
            )
            / n_space_grid
        )

        maximum_scalar_product = tableau[(n_space_grid, n_space_grid)] / n_space_grid

        return (
            norm_squared_initial_srv + norm_squared_end_srv - 2 * maximum_scalar_product
        )

    def _align_single(self, bundle, point, base_point, return_sdist=False):
        r"""Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[k_sampling_points - 1, ambient_dim]
            Point to align.
        base_point : array-like, shape=[k_sampling_points - 1, ambient_dim]
            Reference point.
        return_sdist : bool
            If True, also returns squared distance.

        Returns
        -------
        aligned : array, shape=[k_sampling_points - 1, ambient_dim]
        squared_dist : float
            Quotient distance between point and base point.
            If return_sdist is True.
        """
        n_space_grid = self.n_space_grid
        max_slope = self.max_slope

        k_sampling_points = bundle.total_space.k_sampling_points
        srv_transform = SRVTransform(
            bundle.total_space.ambient_manifold,
            k_sampling_points,
        )
        initial_srv = srv_transform.diffeomorphism(base_point)
        end_srv = srv_transform.diffeomorphism(point)

        initial_srv = self._resample_srv_function(initial_srv, k_sampling_points)
        end_srv = self._resample_srv_function(end_srv, k_sampling_points)

        initial_srv_ = gs.copy(initial_srv)
        end_srv_ = gs.copy(end_srv)

        tableau = (-1.0) * gs.ones((n_space_grid + 1, n_space_grid + 1))
        tableau[0, 0] = 0.0
        gamma = {(0, 0): [(0, 0)]}
        for j in range(1, n_space_grid + 1):
            min_i = int(
                max(
                    gs.floor(j / max_slope),
                    n_space_grid - max_slope * (n_space_grid - j),
                )
            )
            max_i = int(
                min(
                    j * max_slope,
                    gs.ceil(n_space_grid - (n_space_grid - j) * (1 / max_slope)),
                )
            )
            for i in range(min_i, max_i + 1):
                minimum_column_index = int(max(0, i - max_slope))
                minimum_line_index = int(max(0, j - max_slope))
                for m in range(minimum_column_index, i):
                    for k in range(minimum_line_index, j):
                        if tableau[k, m] != -1:
                            new_value = tableau[
                                k, m
                            ] + self._compute_integral_restricted(
                                initial_srv, end_srv, m, i, k, j
                            )

                            if tableau[j, i] < new_value:
                                tableau[j, i] = new_value
                                new_gamma = copy.deepcopy(gamma[(m, k)])
                                new_gamma.append((i, j))
                                gamma[(i, j)] = new_gamma

        point_with_origin = insert_zeros(point, axis=-2)
        point_reparametrized = self._reparametrize(
            point_with_origin, gamma[(n_space_grid, n_space_grid)]
        )[1:]

        if not return_sdist:
            return point_reparametrized

        return point_reparametrized, self._compute_squared_dist(
            initial_srv_, end_srv_, tableau
        )

    def align(self, bundle, point, base_point, return_sdist=False):
        """Vectorize the dynamic programming algorithm.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Initial discrete curve.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            End discrete curve.
        return_sdist : bool
            If True, also returns squared distance.

        Returns
        -------
        aligned : array-like, shape=[..., k_sampling_points - 1, ambient_dim
            Action of the optimal g on point.
        squared_dist : array, shape=[...,]
            Quotient distance between point and base point.
            If return_sdist is True.
        """
        is_batch = check_is_batch(
            bundle.total_space.point_ndim,
            point,
            base_point,
        )
        if not is_batch:
            return self._align_single(
                bundle, point, base_point, return_sdist=return_sdist
            )

        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        out = [
            self._align_single(bundle, point_, base_point_, return_sdist=return_sdist)
            for point_, base_point_ in zip(point, base_point)
        ]
        if not return_sdist:
            return gs.stack(out)

        aligned = gs.stack([out_[0] for out_ in out])
        sdists = gs.stack([out_[1] for out_ in out])
        return aligned, sdists


class SRVTranslationReparametrizationBundle(FiberBundle):
    """Principal bundle of shapes of curves induced by the SRV metric.

    The space of parameterized curves is the total space of a principal
    bundle where the group action is given by reparameterization and the
    base space is the shape space of curves modulo reparametrization, i.e.
    unparametrized curves. In the discrete case, reparametrization corresponds
    to resampling.

    Each tangent vector to the space of parameterized curves can be
    split into a vertical part (tangent to the fibers of the principal
    bundle) and a horizontal part (orthogonal to the vertical part with
    respect to the SRV metric). Horizontal geodesics in the total space
    can be computed using an algorithm that iteratively finds the best
    correspondence between two fibers of the principal bundle, see Reference
    below.

    Parameters
    ----------
    ambient_manifold : Manifold
        Manifold in which curves take values.
    k_sampling_points : int
        Number of sampling points for the discrete curves.

    References
    ----------
    .. [LAB2017] A. Le Brigant,
        "A discrete framework to find the optimal matching between manifold-
        valued curves," in Journal of Mathematical Imaging and Vision, 61,
        pp. 40-70, 2019.
    """

    def __init__(self, total_space):
        super().__init__(total_space=total_space)
        self.aligner = IterativeHorizontalGeodesicAligner()

    def vertical_projection(self, tangent_vec, base_point, return_norm=False):
        """Compute vertical part of tangent vector at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Tangent vector to decompose into horizontal and vertical parts.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve, base point of tangent_vec in the manifold of curves.
        return_norm : boolean,
            If True, the method returns the pointwise norm of the vertical
            part of tangent_vec.

        Returns
        -------
        tangent_vec_ver : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Vertical part of tangent_vec.
        vertical_norm: array-like, shape=[..., n_points]
            Pointwise norm of the vertical part of tangent_vec.
            Only returned when return_norm is True.
        """
        ambient_manifold = self.total_space.ambient_manifold

        a_param, b_param = 1, 1 / 2
        squotient = (a_param / b_param) ** 2

        ndim = self.total_space.point_ndim
        tangent_vec_with_zeros = insert_zeros(tangent_vec, axis=-ndim)
        base_point_with_origin = insert_zeros(base_point, axis=-ndim)

        position = base_point_with_origin[..., 1:-1, :]
        # TODO: why delta=1?
        delta = 1.0
        d_pos = centered_difference(base_point_with_origin, delta=delta, axis=-ndim)
        d_vec = centered_difference(tangent_vec_with_zeros, delta=delta, axis=-ndim)
        d2_pos = second_centered_difference(
            base_point_with_origin, delta=delta, axis=-ndim
        )
        d2_vec = second_centered_difference(
            tangent_vec_with_zeros, delta=delta, axis=-ndim
        )

        pointwise_snorm = ambient_manifold.metric.squared_norm(
            d_pos,
            position,
        )
        pointwise_norm = gs.sqrt(pointwise_snorm)
        pointwise_snorm2 = ambient_manifold.metric.squared_norm(
            d2_pos,
            position,
        )
        pointwise_inner_product_pos = ambient_manifold.metric.inner_product(
            d2_pos, d_pos, position
        )

        vec_a = pointwise_snorm - 1 / 2 * pointwise_inner_product_pos
        vec_b = -2 * pointwise_snorm - squotient * (
            pointwise_snorm2 - pointwise_inner_product_pos**2 / pointwise_snorm
        )
        vec_c = pointwise_snorm + 1 / 2 * pointwise_inner_product_pos
        vec_d = pointwise_norm * (
            ambient_manifold.metric.inner_product(d2_vec, d_pos, position)
            - (squotient - 1)
            * ambient_manifold.metric.inner_product(d_vec, d2_pos, position)
            + (squotient - 2)
            * ambient_manifold.metric.inner_product(d2_pos, d_pos, position)
            * ambient_manifold.metric.inner_product(d_vec, d_pos, position)
            / pointwise_snorm
        )

        linear_system = (
            from_vector_to_diagonal_matrix(vec_a[..., :-1], 1)
            + from_vector_to_diagonal_matrix(vec_b, 0)
            + from_vector_to_diagonal_matrix(vec_c[..., 1:], -1)
        )
        if linear_system.ndim == 2 and tangent_vec.ndim > 2:
            linear_system = gs.broadcast_to(
                linear_system, vec_d.shape[:-1] + linear_system.shape
            )

        vertical_norm = gs.linalg.solve(linear_system, vec_d)

        unit_speed = gs.einsum(
            "...ij,...i->...ij",
            d_pos,
            1 / pointwise_norm,
        )
        tangent_vec_ver = gs.einsum("...ij,...i->...ij", unit_speed, vertical_norm)

        tangent_vec_ver = insert_zeros(tangent_vec_ver, axis=-ndim, end=True)
        if return_norm:
            vertical_norm = insert_zeros(vertical_norm, axis=-1, end=True)
            return tangent_vec_ver, vertical_norm

        return tangent_vec_ver

    def horizontal_projection(self, tangent_vec, base_point):
        """Compute horizontal part of tangent vector at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points, ambient_dim]
            Tangent vector to decompose into horizontal and vertical parts.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve, base point of tangent_vec in the manifold of curves.

        Returns
        -------
        tangent_vec_hor : array-like, shape=[..., k_sampling_points, ambient_dim]
            Horizontal part of tangent_vec.
        """
        tangent_vec_ver = self.vertical_projection(tangent_vec, base_point)
        return tangent_vec - tangent_vec_ver

    def align(self, point, base_point):
        """Find optimal reparametrization of curve with respect to base curve.

        The new parametrization of curve is optimal in the sense that it is the
        member of its fiber closest to the base curve with respect to the SRVMetric.
        It is found as the end point of the horizontal geodesic starting at the base
        curve and ending at the fiber of curve.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Point to align.
        base_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Reference point.

        Returns
        -------
        aligned : array-like, shape=[..., k_sampling_points -1 , ambient_dim]
            Optimal reparametrization of the curve represented by point.
        """
        return self.aligner.align(self, point, base_point)


class SRVTranslationReparametrizationQuotientMetric(QuotientMetric):
    """SRV quotient metric on the space of unparametrized curves.

    This is the class for the quotient metric induced by the SRV Metric
    on the shape space of unparametrized curves, i.e. the space of parametrized
    curves quotiented by the group of reparametrizations. In the discrete case,
    reparametrization corresponds to resampling.
    """
