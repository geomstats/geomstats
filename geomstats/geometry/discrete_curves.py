"""Parameterized curves on any given manifold.

Lead author: Alice Le Brigant.
"""

import logging
import math

from scipy.interpolate import CubicSpline, PchipInterpolator

import geomstats.backend as gs
from geomstats.algebra_utils import from_vector_to_diagonal_matrix
from geomstats.geometry.diffeo import AutodiffDiffeo, Diffeo
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.fiber_bundle import AlignerAlgorithm, FiberBundle
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.landmarks import Landmarks
from geomstats.geometry.manifold import register_quotient_structure
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.nfold_manifold import NFoldManifold, NFoldMetric
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.quotient_metric import QuotientMetric
from geomstats.numerics.finite_differences import (
    centered_difference,
    forward_difference,
    second_centered_difference,
)
from geomstats.numerics.interpolation import (
    LinearInterpolator1D,
    UniformUnitIntervalLinearInterpolator,
)
from geomstats.vectorization import check_is_batch, get_batch_shape


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
    return gs.concatenate((first, second), axis=-array_ndim)


class DiscreteCurvesStartingAtOrigin(NFoldManifold):
    r"""Space of discrete curves modulo translations.

    Each individual curve is represented by a 2d-array of shape `[
    k_sampling_points - 1, ambient_dim]`.

    This space corresponds to the space of immersions defined below, i.e. the
    space of smooth functions from an interval I into the ambient Euclidean
    space M, with non-vanishing derivative.

    .. math::
        Imm(I, M)=\{c \in C^{\infty}(I, M) \|c'(s)\|\neq 0 \forall s \in I \},

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

        self._sphere = Hypersphere(dim=ambient_dim - 1)
        self._discrete_curves_with_l2 = None

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

    @property
    def discrete_curves_with_l2(self):
        """Copy of discrete curves with the L^2 metric."""
        if self._discrete_curves_with_l2 is None:
            self._discrete_curves_with_l2 = self.new(equip=False).equip_with_metric(
                L2CurvesMetric
            )
        return self._discrete_curves_with_l2

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return SRVMetric

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

    def interpolate(self, point):
        """Interpolate between the sampling points of a discrete curve.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve starting at the origin.

        Returns
        -------
        spline : function
            Cubic spline that interpolates between the sampling points
            of the discrete curve.
        """
        k_sampling_points = self.k_sampling_points
        t_space = gs.linspace(0.0, 1.0, k_sampling_points)
        point_with_origin = insert_zeros(point, axis=-self.point_ndim)
        is_batch = check_is_batch(self.point_ndim, point)

        def interpolating_curve(t):
            if not is_batch:
                return gs.from_numpy(
                    CubicSpline(t_space, point_with_origin, axis=-self.point_ndim)(t)
                )
            return gs.stack(
                [
                    gs.from_numpy(
                        CubicSpline(t_space, point_with_origin_, axis=-self.point_ndim)(
                            t
                        )
                    )
                    for point_with_origin_ in point_with_origin
                ]
            )

        return interpolating_curve

    def length(self, point):
        """Compute the length of a discrete curve.

        This is the integral of the absolute value of the velocity.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Discrete curve starting at the origin.

        Returns
        -------
        length : array-like, shape=[..., ]
            Length of the discrete curve.
        """
        point_with_origin = self.insert_origin(point)
        velocity = forward_difference(point_with_origin, axis=-self.point_ndim)
        l2_metric = self.discrete_curves_with_l2.metric
        return l2_metric.norm(velocity, point_with_origin[..., :-1, :])

    def normalize(self, point):
        """Rescale discrete curve to have unit length."""
        return gs.einsum("...ij,...->...ij", point, 1 / self.length(point))


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
            c(s) = c(0) + \int_0^s q(u) |q(u)|du

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

        .. math::
            (h, c) -> dQ_c(h) = |c'|^(-1/2) * (h' - 1/2 * <h',v>v)
            v = c'/|c'|

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Tangent vector to curve.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.
        image_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
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
            (c, k) -> h, \text{ where } dQ_c(h)=k \text{ and } h' = |c'| * (k + <k,v> v)

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
            F transform representation of a discrete curve.

        Returns
        -------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Curve starting at the origin retrieved from its square-root velocity.
        """
        f_polar = self._cartesian_to_polar(image_point)
        f_norms = f_polar[..., :, 0]
        f_args = f_polar[..., :, 1]

        dt = 1 / (self.k_sampling_points - 1)

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
        r"""Compute the left Riemann sum approximation of the integral.

        Compute the left Riemann sum approximation of the integral of a
        function func defined on the unit interval, given by sample points
        at regularly spaced times

        .. math::
            t_i = i / k, \\
            i = 0, ..., k - 1

        where :math:`k` is the number of landmarks (last time is missing).

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


class ElasticMetric(PullbackDiffeoMetric):
    r"""Elastic metric on the space of discrete curves.

    Family of elastic metric parametrized by bending and stretching parameters
    a and b.

    For curves in :math:`\mathbb{R}^2`, it pullbacks the L2 metric by
    the F-transform (see [NK2018]_).

    For curves in :math:`\mathbb{R}^d`, it pullbacks a scaled L2 metric by the SRV
    transform (see [BCKKNP2022_). It only works for ratio :math:`a / 2b = 1`.

    When a=1, and b=1/2, it is equivalent to the SRV metric.

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
    .. [BCKKNP2022] Martin Bauer, Nicolas Charon, Eric Klassen, Sebastian Kurtek,
        Tom Needham, and Thomas Pierron.
        “Elastic Metrics on Spaces of Euclidean Curves: Theory and Algorithms.”
        arXiv, September 20, 2022. https://doi.org/10.48550/arXiv.2209.09862.
    """

    def __init__(
        self,
        space,
        a,
        b=None,
    ):
        if b is None:
            b = a / 2

        self.a = a
        self.b = b

        ambient_manifold = space.ambient_manifold
        k_sampling_points = space.k_sampling_points

        image_space = Landmarks(
            ambient_manifold=space.ambient_manifold,
            k_landmarks=k_sampling_points - 1,
            equip=False,
        )
        image_space.equip_with_metric(L2CurvesMetric)

        if ambient_manifold.dim == 2:
            diffeo = FTransform(space.ambient_manifold, k_sampling_points, a, b)
        elif gs.abs(a / (2 * b) - 1.0) < gs.atol:
            diffeo = SRVTransform(ambient_manifold, k_sampling_points)
            if gs.abs(b - 0.5) > gs.atol:
                scale = 4 * b**2
                image_space.equip_with_metric(scale * image_space.metric)
        else:
            raise ValueError(
                "Cannot instantiate elastic metric for ambient dim="
                f"{ambient_manifold.dim}, a={a} and b={b}. "
                "Ratio must be 1 or ambient_dim=2."
            )

        super().__init__(
            space=space,
            diffeo=diffeo,
            image_space=image_space,
            signature=(math.inf, 0, 0),
        )


class SRVMetric(PullbackDiffeoMetric):
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


class IterativeHorizontalGeodesicAligner(AlignerAlgorithm):
    r"""Align two curves through iterative horizontal geodesic algorithm.

    This algorithm computes the horizontal geodesic between two curves in the shape
    bundle of curves modulo reparametrizations, and at the same time, aligns the
    end curve with respect to the initial curve. This is done through an iterative
    procedure where the initial curve stays fixed and the sampling points are moved
    on the end curve to obtain its optimal parametrization with respect to the initial
    curve. This procedure is based on the decomposition of any path of curves into a
    horizontal path of curves composed with a path of reparametrizations:
    :math:`c(t, s) = c_h(t, phi(t, s))` where :math:`d/dt c_h(t, .)` is horizontal.
    Here t is the time parameter of the path and s the space parameter of the curves.

    The algorithm sets current_end_curve to be the end curve and iterates three steps:
    1) compute the geodesic between the initial curve and current_end_curve
    2) compute the path of reparametrizations such that the path of its inverses
    transforms this geodesic into a horizontal path of curves
    3) invert this path of reparametrizations to find the horizontal path and update
    current_end_curve to be its end point.
    The algorithm stops when the new current_end_curve is sufficiently
    close to the former current_end_curve.

    Parameters
    ----------
    total_space : Manifold
        Total space with reparametrizations fiber bundle structure.
    n_time_grid : int
        Number of times in which compute the geodesic.
    threshold : float
        When the difference between the new end curve and the current end
        curve becomes lower than this threshold, the algorithm stops.
        Optional, default: 1e-3.
    max_iter : int
        Maximum number of iterations.
        Optional, default: 20.
    tol: float
        Minimal spacing between time samples in the unit segment when
        reparametrizing the end curve.
        Optional, default: 1e-3.
    verbose: boolean
        Optional, default: False.
    save_history : bool
        If True, history is saved in a `self.history`.

    References
    ----------
    .. [LAB2017] A. Le Brigant, "Optimal matching between curves in a manifold",
        in Geometric Science of Information. Springer Lecture Notes in Computer
        Science 10589 (2017), 57 - 64. https://hal.science/hal-04374199.
    """

    def __init__(
        self,
        total_space,
        n_time_grid=100,
        threshold=1e-3,
        max_iter=20,
        tol=1e-3,
        verbose=0,
        save_history=False,
    ):
        super().__init__(total_space)
        self.n_time_grid = n_time_grid
        self.threshold = threshold
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.save_history = save_history
        if save_history:
            self.history = None

    @staticmethod
    def _euler_step_forward(current, increment, step, tol):
        r"""Perform Euler step while enforcing increasing solution.

        Compute the new state phi(t+dt,.) from the current state phi(t,.) using
        the Euler step: new = current + step * increment, i.e.
        :math:`phi(t+dt,.) = phi(t,.) + dt * d/dt phi(t,.)`
        while ensuring that the result is an increasing function on the unit
        interval that preserves the end points 0 and 1. This is done by alternatively
        computing phi(t+dt,s) and phi(t+dt,1-s), for s increasing from 0 to 0.5,
        in a way that ensures that the spacing with the previous and following
        sampling points are greater than a given tolerance.
        """
        k_sampling_points = current.shape[0]
        max_index = k_sampling_points // 2
        new = gs.copy(current)
        sign = gs.sign(tol)
        for index in range(1, max_index):
            new_index = current[index] + step * increment[index - 1]
            new_index_min = new[index - 1] + tol
            new_index_max = (
                new[k_sampling_points - index] - (k_sampling_points - 2 * index) * tol
            )
            new_index = sign * gs.maximum(sign * new_index, sign * new_index_min)
            new[index] = sign * gs.minimum(sign * new_index, sign * new_index_max)

            symindex = k_sampling_points - 1 - index
            new_symindex = current[symindex] + step * increment[symindex - 1]
            new_symindex_min = new[index] + (k_sampling_points - 2 * index - 1) * tol
            new_symindex_max = new[symindex + 1] - tol
            new_symindex = sign * gs.minimum(
                sign * new_symindex, sign * new_symindex_max
            )
            new[symindex] = sign * gs.maximum(
                sign * new_symindex, sign * new_symindex_min
            )

        if k_sampling_points % 2 > 0:
            new_midindex = current[index + 1] + step * increment[index]
            new_midindex_min = new[index] + tol
            new_midindex_max = new[index + 2] - tol
            new_midindex = sign * gs.maximum(
                sign * new_midindex, sign * new_midindex_min
            )
            new[index + 1] = sign * gs.minimum(
                sign * new_midindex, sign * new_midindex_max
            )

        return new

    def _euler_step(self, current, increment, step):
        r"""Perform Euler step while enforcing increasing solution.

        Symmetric version of previous function.
        """
        return (
            self._euler_step_forward(current, increment, step, self.tol)
            + gs.flip(
                self._euler_step_forward(
                    gs.flip(current, axis=0),
                    gs.flip(increment, axis=0),
                    step,
                    -self.tol,
                ),
                axis=0,
            )
        ) / 2

    def _construct_reparametrization(self, vertical_norm, space_deriv_norm):
        r"""Construct path of reparametrizations.

        Construct path of reparametrizations phi(t, s) in the decomposition
        of the path of curves
        :math:`c(t, s) = c_h(t, phi(t, s))` where :math:`d/dt c_h(t, .)` is horizontal.
        This is done by solving a partial differential equation, using an Euler
        step that enforces that the solution phi(t,.) is an increasing function of
        the unit interval that preserves the end points 0 and 1, for all time t.

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
            Path of parametrizations, such that the path of curves composed
            with the path of inverse parametrizations is a horizontal path.
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
            repar = repar_i = self._euler_step(repar, repar_time_deriv, 1 / n_times)

            repars.append(repar_i)

        return gs.stack(repars)

    def _invert_reparametrize_single(self, t_space, repar, point):
        """Invert path of reparametrizations, non vectorized."""
        spline = CubicSpline(t_space, point, axis=0)
        repar_inverse = PchipInterpolator(repar, t_space)
        return gs.from_numpy(spline(repar_inverse(t_space)))

    def _invert_reparametrization(
        self,
        t_space,
        repar,
        path_of_curves,
        repar_inverse_end,
        end_spline,
    ):
        r"""Invert path of reparametrizations.

        Given a path of curves c(t, s) and a path of reparametrizations
        phi(t, s), compute:
        :math:`c(t, phi_inv(t, s))` where `phi_inv(t, .) = phi(t, .)^{-1}`

        The computation for the last time t=1 is done differently, using
        the spline function associated to the end curve and the composition
        of the inverse reparametrizations contained in rep_inverse_end:
        :math:`end_spline ° phi_inv(1, .) ° ... ° phi_inv(0, .)`.

        Parameters
        ----------
        repar : array-like, shape=[n_times, k_sampling_points]
            Path of reparametrizations.
        path_of_curves : array-like, shape=[n_times, k_sampling_points, ambient_dim]
            Path of curves.
        repar_inverse_end: list
            List of the inverses of the reparametrizations applied to
            the end curve during the optimal matching algorithm.
        end_spline : function
            Spline interpolation of the end point of the path of curves.

        Returns
        -------
        reparametrized_path : array-like, \
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

        repar_inverse_end.append(PchipInterpolator(repar[-1], t_space))
        arg = t_space
        for repar_inverse in reversed(repar_inverse_end):
            arg = repar_inverse(arg)
        end_curve_repar = end_spline(arg)

        reparametrized_path.append(end_curve_repar)
        return gs.stack(reparametrized_path)

    def _iterate(
        self,
        times,
        t_space,
        initial_point,
        end_point,
        repar_inverse_end,
        end_spline,
    ):
        """Perform one step of the alignment algorithm."""
        ndim = self._total_space.point_ndim

        total_space_geod_fun = self._total_space.metric.geodesic(
            initial_point=initial_point, end_point=end_point
        )
        geod_points = total_space_geod_fun(times)
        geod_points_with_origin = insert_zeros(geod_points, axis=-ndim)

        time_deriv = forward_difference(geod_points, axis=-(ndim + 1))
        _, vertical_norm = self._total_space.fiber_bundle.vertical_projection(
            time_deriv, geod_points[:-1], return_norm=True
        )
        vertical_norm = insert_zeros(vertical_norm, axis=-1)

        space_deriv = centered_difference(
            geod_points_with_origin, axis=-ndim, endpoints=True
        )[:-1]

        pointwise_space_deriv_norm = self._total_space.ambient_manifold.metric.norm(
            space_deriv, geod_points_with_origin[:-1]
        )

        repar = self._construct_reparametrization(
            vertical_norm,
            pointwise_space_deriv_norm,
        )

        horizontal_path = self._invert_reparametrization(
            t_space, repar, geod_points_with_origin, repar_inverse_end, end_spline
        )

        return horizontal_path

    def _discrete_horizontal_geodesic_single(
        self, initial_point, end_point, end_spline
    ):
        """Compute discrete horizontal geodesic, non vectorized.

        Parameters
        ----------
        initial_point : array-like, shape=[k_sampling_points, ambient_dim]
            Initial discrete curve.
        end_point : array-like, shape=[k_sampling_points, ambient_dim]
            End discrete curve.
        end_spline : function
            Spline interpolation of end point.

        Returns
        -------
        horizontal_geod_points : array, shape=[n_time_grid, k - 1, ambient_dim]
            Geodesic points.
        """
        times = gs.linspace(0.0, 1.0, self.n_time_grid)

        k_sampling_points = self._total_space.k_sampling_points
        t_space = gs.linspace(0.0, 1.0, k_sampling_points)

        current_end_point = end_point
        repar_inverse_end = []
        for index in range(self.max_iter):
            horizontal_path_with_origin = self._iterate(
                times,
                t_space,
                initial_point,
                current_end_point,
                repar_inverse_end,
                end_spline,
            )
            new_end_point = horizontal_path_with_origin[-1][1:]
            l2_metric = self._total_space.discrete_curves_with_l2.metric
            gap = l2_metric.dist(new_end_point, current_end_point)
            current_end_point = new_end_point

            if gap < self.threshold:
                if self.verbose > 0:
                    logging.info(
                        f"Convergence of alignment reached after {index + 1} "
                        "iterations."
                    )
                break
        else:
            logging.warning(
                "Maximum number of iterations %d reached. The result may be inaccurate",
                self.max_iter,
            )

        if self.save_history:
            self.history = dict(
                spline=end_spline,
                repar_inverse=repar_inverse_end,
            )

        return horizontal_path_with_origin[..., 1:, :]

    def discrete_horizontal_geodesic(self, initial_point, end_point, end_spline):
        """Compute discrete horizontal geodesic.

        Parameters
        ----------
        initial_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            Initial discrete curve.
        end_point : array-like, shape=[..., k_sampling_points, ambient_dim]
            End discrete curve.
        end_spline : callable or list[callable]
            Spline interpolation of end point.

        Returns
        -------
        geod_points : array, shape=[..., n_time_grid, k - 1, ambient_dim]
        """
        is_batch = check_is_batch(
            self._total_space.point_ndim,
            initial_point,
            end_point,
        )
        if not is_batch:
            return self._discrete_horizontal_geodesic_single(
                initial_point, end_point, end_spline
            )

        if initial_point.ndim != end_point.ndim:
            initial_point, end_point = gs.broadcast_arrays(initial_point, end_point)

        return gs.stack(
            [
                self._discrete_horizontal_geodesic_single(
                    initial_point_, end_point_, end_spline_
                )
                for initial_point_, end_point_, end_spline_ in zip(
                    initial_point, end_point, end_spline
                )
            ]
        )

    def align(self, point, base_point):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve to align.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Reference discrete curve.

        Returns
        -------
        aligned : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Curve reparametrized in an optimal way with respect to reference curve.
        """
        if point.ndim == self._total_space.point_ndim:
            spline = self._total_space.interpolate(point)
            if base_point.ndim > self._total_space.point_ndim:
                spline = [spline] * base_point.shape[0]
        else:
            spline = [self._total_space.interpolate(point_) for point_ in point]

        return self.discrete_horizontal_geodesic(base_point, point, spline)[
            ..., -1, :, :
        ]


class DynamicProgrammingAligner(AlignerAlgorithm):
    r"""Align two curves through dynamic programming.

    Find the reparametrization gamma of end_curve that minimizes the distance
    between initial_curve and end_curve reparametrized by gamma, and output
    the corresponding distance, using a dynamic programming algorithm.

    The objective can be expressed in terms of square root velocity (SRV)
    representations: it is equivalent to finding the gamma that maximizes
    the L2 scalar product between :math:`initial_{srv}` and :math:`end_{srv}@\gamma`,
    where :math:`initial_{srv}` is the SRV representation of the initial curve
    and :math:`end_{srv}@\gamma` is the SRV representation of the end curve
    reparametrized by :math:`\gamma`, i.e

    .. math::
        end_{srv}@\gamma(t) = end_{srv}(\gamma(t))\cdot|\gamma(t)|^\frac{1}{2}

    The dynamic programming algorithm assumes that for every subinterval
    :math:`\left[\frac{i}{n},\frac{i+1}{n}\right]` of :math:`\left[0,1\right]`,
    gamma is linear.

    Parameters
    ----------
    total_space : Manifold
        Total space with reparametrizations fiber bundle structure.
    n_space_grid : int
        Number of sampling points of the unit interval.
    max_slope : int
        Maximum slope allowed for a reparametrization.

    References
    ----------
    [WAJ2007] M. Washington, S. Anuj & H. Joshi,
        "On Shape of Plane Elastic Curves", in International Journal of Computer
        Vision. 73(3):307-324, 2007.
    """

    def __init__(self, total_space, n_space_grid=100, max_slope=6):
        k_sampling_points = total_space.k_sampling_points

        super().__init__(total_space)
        self.n_space_grid = n_space_grid
        self.max_slope = max_slope

        self._srv_transform = SRVTransform(
            self._total_space.ambient_manifold,
            k_sampling_points,
        )

    def _resample_srv_function(self, srv_function):
        """Resample SRV function of a discrete curve.

        Parameters
        ----------
        srv_function : array, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve.

        Returns
        -------
        srv : array, shape=[..., n_space_grid - 1, ambient_dim]
            SRV function of the curve at the right size.

        Notes
        -----
        If `n_space_grid` < `k_sampling_points`, it subsamples the curve.
        """
        n_space_grid = self.n_space_grid
        index = gs.array(range(n_space_grid - 1))

        ratio = (self._total_space.k_sampling_points - 1) / (n_space_grid - 1)
        indices = gs.cast(gs.floor(index * ratio), dtype=int)

        return srv_function[..., indices, :]

    @staticmethod
    def _compute_restricted_integral(
        srv_1, srv_2, x_min_index, x_max_index, y_min_index, y_max_index
    ):
        r"""Compute the value of an integral over a subinterval.

        Compute n * the value of the integral of

        .. math::

            q_1(t)\cdot q_2(\gamma(t))\cdot|\gamma(t)|^\frac{1}{2}

        over :math:`\left[\x_min,x_max\right]` where gamma restricted to
        :math:`\left[\x_min,x_max\right]` is a linear.

        Parameters
        ----------
        srv_1 : array, shape=[n, ambient_dim]
            SRV function of the initial curve.
        srv_2 : array, shape=[n, ambient_dim]
            SRV function of the end curve.
        x_min_index : int
            Beginning of the subinterval.
        x_max_index : int
            End of the subinterval.
        y_min_index : int
            Value of gamma at x_min.
        y_max_index : int
            Value of gamma at x_max.

        Returns
        -------
        value : float
            Value of the integral described above.
        """
        delta_y_index = y_max_index - y_min_index
        delta_x_index = x_max_index - x_min_index

        gamma_slope = delta_y_index / delta_x_index

        x_bound_indices = list(range(x_min_index, x_max_index + 1))
        y_bounds_indices = [
            (y_index - y_min_index) / gamma_slope + x_min_index
            for y_index in range(y_min_index, y_max_index + 1)
        ]

        lower_bound = x_min_index
        x_index, y_index = 0, 0

        value = 0.0
        while x_index < delta_x_index and y_index < delta_y_index:
            upper_bound = min(
                x_bound_indices[x_index + 1], y_bounds_indices[y_index + 1]
            )

            # NB: normalization is done outside
            length = upper_bound - lower_bound
            value += length * gs.dot(
                srv_1[x_min_index + x_index], srv_2[y_min_index + y_index]
            )

            if (
                gs.abs(x_bound_indices[x_index + 1] - y_bounds_indices[y_index + 1])
                < gs.atol
            ):
                x_index += 1
                y_index += 1
            elif x_bound_indices[x_index + 1] < y_bounds_indices[y_index + 1]:
                x_index += 1
            else:
                y_index += 1
            lower_bound = upper_bound

        return gamma_slope**0.5 * value

    def _reparametrize(self, point_with_origin, gamma):
        """Reparametrize curve by gamma.

        Parameters
        ----------
        point_with_origin : array, shape=[k_sampling_points, ambient_dim]
            Discrete curve.
        gamma : array, shape=[n_subinterval]
            Parametrization of a curve.

        Returns
        -------
        new_point : array , shape=[k_sampling_points - 1, ambient_dim]
            Curve reparametrized by gamma.
        """
        n_space_grid = self.n_space_grid
        k_sampling_points = self._total_space.k_sampling_points

        x_uniform = gs.linspace(0.0, 1.0, k_sampling_points)
        point_interpolator = UniformUnitIntervalLinearInterpolator(
            point_with_origin, point_ndim=1
        )

        x_space = gs.array([gamma_elem[1] for gamma_elem in gamma]) / (n_space_grid - 1)
        y_space = gs.array([gamma_elem[0] for gamma_elem in gamma]) / (n_space_grid - 1)

        repar_inverse = LinearInterpolator1D(y_space, x_space, point_ndim=0)

        return point_interpolator(repar_inverse(x_uniform[1:]))

    def _compute_squared_dist(self, initial_srv, end_srv, grid_last):
        """Compute squared distance using algorithmic information."""
        n_space_grid = self.n_space_grid

        norm_squared_initial_srv = self._compute_restricted_integral(
            initial_srv, initial_srv, 0, n_space_grid - 1, 0, n_space_grid - 1
        ) / (n_space_grid - 1)
        norm_squared_end_srv = self._compute_restricted_integral(
            end_srv, end_srv, 0, n_space_grid - 1, 0, n_space_grid - 1
        ) / (n_space_grid - 1)

        maximum_scalar_product = grid_last / (n_space_grid - 1)
        return (
            norm_squared_initial_srv + norm_squared_end_srv - 2 * maximum_scalar_product
        )

    def _align_single(self, point, base_point, return_sdist=False):
        r"""Align point to base point, non vectorized.

        Parameters
        ----------
        point : array-like, shape=[k_sampling_points - 1, ambient_dim]
            Discrete curve to align.
        base_point : array-like, shape=[k_sampling_points - 1, ambient_dim]
            Reference discrete curve.
        return_sdist : bool
            If True, also returns squared distance.

        Returns
        -------
        aligned : array, shape=[k_sampling_points - 1, ambient_dim]
            Curve reparametrized in an optimal way with respect to reference curve.
        squared_dist : float
            Quotient distance between point and base point.
            If return_sdist is True.
        """
        max_slope = self.max_slope
        n_space_grid = self.n_space_grid

        initial_srv = self._srv_transform.diffeomorphism(base_point)
        end_srv = self._srv_transform.diffeomorphism(point)

        resampled_initial_srv = self._resample_srv_function(initial_srv)
        resampled_end_srv = self._resample_srv_function(end_srv)

        grid = {(0, 0): 0.0}
        gamma = {(0, 0): [(0, 0)]}

        for x_max_index in range(1, n_space_grid):
            max_y_max_index = max(n_space_grid, x_max_index + 1)
            for y_max_index in range(1, max_y_max_index):
                # NB: solves minimization part
                min_x_min_index = max(0, x_max_index - max_slope)
                min_y_min_index = max(0, y_max_index - max_slope)
                for x_min_index in range(min_x_min_index, x_max_index):
                    for y_min_index in range(min_y_min_index, y_max_index):
                        if (x_min_index, y_min_index) not in grid:
                            # jump borders
                            continue
                        elif (
                            gs.abs(
                                (y_max_index - y_min_index)
                                / (x_max_index - x_min_index)
                                - 1
                            )
                            < gs.atol
                            and x_max_index - x_min_index != 1
                            and y_max_index - y_max_index != 1
                        ):
                            # jump slope == 1 except last
                            continue

                        new_value = grid[
                            x_min_index, y_min_index
                        ] + self._compute_restricted_integral(
                            resampled_initial_srv,
                            resampled_end_srv,
                            x_min_index,
                            x_max_index,
                            y_min_index,
                            y_max_index,
                        )
                        if (x_max_index, y_max_index) not in grid or grid[
                            (x_max_index, y_max_index)
                        ] < new_value:
                            grid[x_max_index, y_max_index] = new_value
                            new_gamma = gamma[(x_min_index, y_min_index)].copy()
                            new_gamma.append((x_max_index, y_max_index))
                            gamma[(x_max_index, y_max_index)] = new_gamma

        last_index = n_space_grid - 1
        point_with_origin = insert_zeros(point, axis=-2)
        point_reparametrized = self._reparametrize(
            point_with_origin, gamma[last_index, last_index]
        )

        if not return_sdist:
            return point_reparametrized

        return point_reparametrized, self._compute_squared_dist(
            resampled_initial_srv,
            resampled_end_srv,
            grid[(last_index, last_index)],
        )

    def align(self, point, base_point, return_sdist=False):
        """Align point to base point.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve to align.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Reference discrete curve.
        return_sdist : bool
            If True, also returns squared distance.

        Returns
        -------
        aligned : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Curve reparametrized in an optimal way with respect to reference curve.
        squared_dist : array, shape=[...,]
            Quotient distance between point and base point.
            If return_sdist is True.
        """
        is_batch = check_is_batch(
            self._total_space.point_ndim,
            point,
            base_point,
        )
        if not is_batch:
            return self._align_single(point, base_point, return_sdist=return_sdist)

        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        out = [
            self._align_single(point_, base_point_, return_sdist=return_sdist)
            for point_, base_point_ in zip(point, base_point)
        ]
        if not return_sdist:
            return gs.stack(out)

        aligned = gs.stack([out_[0] for out_ in out])
        sdists = gs.stack([out_[1] for out_ in out])
        return aligned, sdists


class SRVReparametrizationBundle(FiberBundle):
    """Principal bundle of curves modulo reparameterizations with the SRV metric.

    The space of parameterized curves is the total space of a principal bundle
    where the group action is given by reparameterization and the base space is
    the shape space of curves modulo reparametrization, i.e.unparametrized
    curves. In the discrete case, reparametrization corresponds to resampling.

    Each tangent vector to the space of parameterized curves can be split into a
    vertical part (tangent to the fibers of the principal bundle) and a
    horizontal part (orthogonal to the vertical part with respect to the SRV
    metric). The geodesic between the shapes of two curves is computed by
    aligning (i.e. reparametrizing) one of the two curves with respect to the
    other, and computing the geodesic between the aligned curves. This geodesic
    will be horizontal, and will project to a geodesic on the shape space.

    Two different aligners are available:
    - IterativeHorizontalGeodesicAligner (default)
    - DynamicProgrammingAligner.

    Parameters
    ----------
    total_space : DiscreteCurvesStartingAtOrigin
        Space of discrete curves starting at the origin
    """

    def __init__(self, total_space, aligner=None):
        if aligner is None:
            aligner = IterativeHorizontalGeodesicAligner(total_space)

        super().__init__(
            total_space=total_space,
            aligner=aligner,
        )

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
        ambient_manifold = self._total_space.ambient_manifold

        a_param, b_param = 1, 1 / 2
        squotient = (a_param / b_param) ** 2

        ndim = self._total_space.point_ndim
        tangent_vec_with_zeros = insert_zeros(tangent_vec, axis=-ndim)
        base_point_with_origin = insert_zeros(base_point, axis=-ndim)

        position = base_point_with_origin[..., 1:-1, :]
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


class SRVRotationBundle(FiberBundle):
    """Principal bundle of curves modulo rotations with the SRV metric.

    This is the fiber bundle where the total space is the space of parameterized
    curves equipped with the SRV metric, the action is given by rotations, and
    the base space is the shape space of curves modulo rotations.

    Parameters
    ----------
    total_space : DiscreteCurvesStartingAtOrigin
        Space of discrete curves starting at the origin
    """

    def _rotate(self, point, rotation):
        """Rotate discrete curve starting at origin."""
        return Matrices.transpose(gs.matmul(rotation, Matrices.transpose(point)))

    def align(self, point, base_point, return_rotation=False):
        """Align point to base point.

        Find optimal rotation of curve with respect to base curve.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve to align.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Reference discrete curve.
        return_rotation : boolean
            If true, returns the optimal rotation used for the alignment.
            Optional, default : False.

        Returns
        -------
        aligned : array-like, shape=[..., k_sampling_points - 1, ambient_dim
            Curve optimally rotated with respect to reference curve.
        """
        srv_transform = self._total_space.metric.diffeo
        initial_srv = srv_transform.diffeomorphism(base_point)
        end_srv = srv_transform.diffeomorphism(point)

        mat = gs.matmul(Matrices.transpose(initial_srv), end_srv)
        u_svd, _, vt_svd = gs.linalg.svd(mat)
        sign = gs.linalg.det(gs.matmul(u_svd, vt_svd))
        vt_svd[..., -1, :] = gs.einsum("...,...j->...j", sign, vt_svd[..., -1, :])
        rotation = gs.matmul(u_svd, vt_svd)

        point_aligned = self._rotate(point, rotation)
        if return_rotation:
            return point_aligned, rotation

        return point_aligned


class SRVRotationReparametrizationBundle(FiberBundle):
    """SRV principal bundle of curves modulo rotations and reparametrizations.

    This is the fiber bundle where the total space is the space of parameterized
    curves equipped with the SRV metric, the action is the joint action of rotations
    and reparametrizations, and the base space is the shape space of curves modulo
    rotations and reparametrizations.

    Parameters
    ----------
    total_space : DiscreteCurvesStartingAtOrigin
        Space of discrete curves starting at the origin
    threshold : float
        Parameter used in the alignment of a curve with respect to a base curve.
        When the difference between the new curve and the current curve becomes lower
        than this threshold, the alignment algorithm stops.
        Optional, default: 1e-3.
    max_iter : int
        Maximum number of iterations in the alignment of a curve with respect to a base.
        curve.
        Optional, default: 20.
    verbose : boolean
        Parameter used in the alignment of a curve with respect to a base curve.
        Optional, default: False.
    """

    def __init__(self, total_space, threshold=1e-3, max_iter=20, verbose=0):
        super().__init__(total_space=total_space)

        self.threshold = threshold
        self.max_iter = max_iter
        self.verbose = verbose
        self._total_space_with_rotations = self._init_space_with_rotations(total_space)
        self._total_space_with_reparametrizations = (
            self._init_space_with_reparametrizations(total_space)
        )

    def _init_space_with_rotations(self, total_space):
        space = total_space.new(equip=True)
        space.equip_with_group_action("rotations")
        space.fiber_bundle = SRVRotationBundle(space)
        return space

    def _init_space_with_reparametrizations(self, total_space):
        space = total_space.new(equip=True)
        space.equip_with_group_action("reparametrizations")
        space.fiber_bundle = SRVReparametrizationBundle(space)
        return space

    def _align_single(self, point, base_point):
        """Align point to base point.

        This is achieved by iteratively rotating and reparametrizing the curve to align
        with respect to the reference curve until convergence.

        Parameters
        ----------
        point : array-like, shape=[k_sampling_points - 1, ambient_dim]
            Discrete curve to align.
        base_point : array-like, shape=[k_sampling_points - 1, ambient_dim]
            Reference discrete curve.

        Returns
        -------
        aligned : array-like, shape=[k_sampling_points - 1, ambient_dim]
             Curve rotated and reparametrized in an optimal way with respect to
             the reference curve.
        """
        aligned_point = previous_aligned_point = point
        for index in range(self.max_iter):
            for total_space in (
                self._total_space_with_rotations,
                self._total_space_with_reparametrizations,
            ):
                aligned_point = total_space.fiber_bundle.align(
                    aligned_point, base_point
                )

            l2_metric = self._total_space.discrete_curves_with_l2.metric
            gap = l2_metric.dist(aligned_point, previous_aligned_point)
            previous_aligned_point = aligned_point

            if gap < self.threshold:
                if self.verbose > 0:
                    logging.info(
                        f"Convergence of alignment reached after {index + 1} "
                        "iterations."
                    )

                break
        else:
            logging.warning(
                f"Maximum number of iterations {self.max_iter} reached during "
                "alignment with respect to rotations and reparametrizations. "
                "The result may be inaccurate."
            )
        return aligned_point

    def align(self, point, base_point):
        """Align point to base point.

        This is achieved by iteratively rotating and reparametrizing the curve to align
        with respect to the reference curve until convergence.

        Parameters
        ----------
        point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Discrete curve to align.
        base_point : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
            Reference discrete curve.

        Returns
        -------
        aligned : array-like, shape=[..., k_sampling_points - 1, ambient_dim]
             Curve rotated and reparametrized in an optimal way with respect to
             the reference curve.
        """
        is_batch = check_is_batch(
            self._total_space.point_ndim,
            point,
            base_point,
        )
        if not is_batch:
            return self._align_single(point, base_point)

        if point.ndim != base_point.ndim:
            point, base_point = gs.broadcast_arrays(point, base_point)

        return gs.stack(
            [
                self._align_single(point_, base_point_)
                for point_, base_point_ in zip(point, base_point)
            ]
        )


register_quotient_structure(
    DiscreteCurvesStartingAtOrigin,
    SRVMetric,
    "rotations",
    SRVRotationBundle,
    QuotientMetric,
)
register_quotient_structure(
    DiscreteCurvesStartingAtOrigin,
    SRVMetric,
    "reparametrizations",
    SRVReparametrizationBundle,
    QuotientMetric,
)
register_quotient_structure(
    DiscreteCurvesStartingAtOrigin,
    SRVMetric,
    "rotations and reparametrizations",
    SRVRotationReparametrizationBundle,
    QuotientMetric,
)
