"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded with
the hyperboloid representation (embedded in minkowsky space).
"""

import math

import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hyperbolic import HyperbolicMetric
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.minkowski import MinkowskiMetric

TOLERANCE = 1e-6

SINH_TAYLOR_COEFFS = [0., 1.,
                      0., 1 / math.factorial(3),
                      0., 1 / math.factorial(5),
                      0., 1 / math.factorial(7),
                      0., 1 / math.factorial(9)]
COSH_TAYLOR_COEFFS = [1., 0.,
                      1 / math.factorial(2), 0.,
                      1 / math.factorial(4), 0.,
                      1 / math.factorial(6), 0.,
                      1 / math.factorial(8), 0.]
INV_SINH_TAYLOR_COEFFS = [0., - 1. / 6.,
                          0., + 7. / 360.,
                          0., - 31. / 15120.,
                          0., + 127. / 604800.]
INV_TANH_TAYLOR_COEFFS = [0., + 1. / 3.,
                          0., - 1. / 45.,
                          0., + 2. / 945.,
                          0., -1. / 4725.]

EPSILON = 1e-6


class Hyperboloid(Hyperbolic, EmbeddedManifold):
    """Class for the n-dimensional hyperbolic space.

    Class for the n-dimensional hyperbolic space
    as embedded in (n+1)-dimensional Minkowski space.

    The coords_type parameter allows to choose the
    representation of the points as input.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    coords_type : str, {'extrinsic', 'intrinsic'}
        Default coordinates to represent points in hyperbolic space.
        Optional, default: 'extrinsic'.
    scale : int
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
        Optional, default: 1.
    """

    default_coords_type = 'extrinsic'
    default_point_type = 'vector'

    def __init__(self, dim, coords_type='extrinsic', scale=1):
        # TODO (ninamiolane): Call __init__ from parent classes
        # and remove ignore rule of corresponding DeepSource issue
        self.scale = scale
        self.dim = dim
        self.coords_type = coords_type
        self.point_type = Hyperboloid.default_point_type
        self.embedding_manifold = Minkowski(dim + 1)
        self.embedding_metric = self.embedding_manifold.metric
        self.metric =\
            HyperboloidMetric(self.dim, self.coords_type, self.scale)

    def belongs(self, point, tolerance=TOLERANCE):
        """Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space in
        its hyperboloid representation.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be tested.
        tolerance : float, optional
            Tolerance at which to evaluate how close the squared norm
            is to the reference value.
            Optional, default: 1e-6.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Array of booleans indicating whether the corresponding points
            belong to the hyperbolic space.
        """
        point_dim = point.shape[-1]
        if point_dim is not self.dim + 1:
            belongs = False
            if point_dim is self.dim and self.coords_type == 'intrinsic':
                belongs = True
            if gs.ndim(point) == 2:
                belongs = gs.tile([belongs], (point.shape[0],))
            return belongs

        sq_norm = self.embedding_metric.squared_norm(point)
        euclidean_sq_norm = gs.linalg.norm(point, axis=-1) ** 2
        diff = gs.abs(sq_norm + 1)
        belongs = diff < tolerance * euclidean_sq_norm
        return belongs

    def regularize(self, point):
        """Regularize a point to the canonical representation.

        Regularize a point to the canonical representation chosen
        for the hyperbolic space, to avoid numerical issues.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space in canonical representation
            in extrinsic coordinates.
        """
        if self.coords_type == 'intrinsic':
            point = self.intrinsic_to_extrinsic_coords(point)

        point = gs.to_ndarray(point, to_ndim=2)

        sq_norm = self.embedding_metric.squared_norm(point)
        real_norm = gs.sqrt(gs.abs(sq_norm))

        mask_0 = gs.isclose(real_norm, 0.)
        mask_not_0 = ~mask_0
        mask_not_0_float = gs.cast(mask_not_0, gs.float32)
        projected_point = point

        normalized_point = gs.einsum(
            '...,...i->...i', 1. / real_norm, point)
        projected_point = gs.einsum(
            '...,...i->...i', mask_not_0_float, normalized_point)
        return projected_point

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the hyperbolic space.

        Project a vector in Minkowski space on the tangent space
        of the hyperbolic space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector in Minkowski space to be projected.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at the base point, equal to the projection of
            the vector in Minkowski space.
        """
        if self.coords_type == 'intrinsic':
            base_point = self.intrinsic_to_extrinsic_coords(base_point)

        sq_norm = self.embedding_metric.squared_norm(base_point)
        inner_prod = self.embedding_metric.inner_product(
            base_point, vector)

        coef = inner_prod / sq_norm

        tangent_vec = vector - gs.einsum('...,...j->...j', coef, base_point)
        return tangent_vec

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert from intrinsic to extrinsic coordinates.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., dim]
            Point in the embedded manifold in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in the embedded manifold in extrinsic coordinates.
        """
        if self.dim != point_intrinsic.shape[-1]:
            raise NameError("Wrong intrinsic dimension: "
                            + str(point_intrinsic.shape[-1]) + " instead of "
                            + str(self.dim))
        return Hyperbolic.change_coordinates_system(
            point_intrinsic, 'intrinsic', 'extrinsic')

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert from extrinsic to intrinsic coordinates.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point in the embedded manifold in extrinsic coordinates,
            i. e. in the coordinates of the embedding manifold.

        Returns
        -------
        point_intrinsic : array-like, shape=[..., dim]
            Point in intrinsic coordinates.
        """
        belong_point = self.belongs(point_extrinsic)
        if not gs.all(belong_point):
            raise NameError("Point do not belong to the hyperboloid")
        return\
            Hyperbolic.change_coordinates_system(point_extrinsic,
                                                 'extrinsic',
                                                 'intrinsic')


class HyperboloidMetric(HyperbolicMetric):
    """Class that defines operations using a hyperbolic metric.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    point_type : str, {'extrinsic', 'intrinsic', etc}
        Default coordinates to represent points in hyperbolic space.
        Optional, default: 'extrinsic'.
    scale : int
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
        Optional, default: 1.
    """

    default_point_type = 'vector'
    default_coords_type = 'extrinsic'

    def __init__(self, dim, coords_type='extrinsic', scale=1):
        super(HyperboloidMetric, self).__init__(
            dim=dim,
            scale=scale)
        self.embedding_metric = MinkowskiMetric(dim + 1)

        self.coords_type = coords_type
        self.point_type = HyperbolicMetric.default_point_type

        self.scale = scale

    def inner_product_matrix(self, base_point=None):
        """Compute the inner product matrix.

        Parameters
        ----------
        base_point: array-like, shape=[..., dim + 1]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_prod_mat: array-like, shape=[..., dim+1, dim + 1]
            Inner-product matrix.
        """
        self.embedding_metric.inner_product_matrix(base_point)

    def _inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1], optional
            Point in hyperbolic space.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        inner_prod = self.embedding_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point)
        return inner_prod

    def _squared_norm(self, vector, base_point=None):
        """Compute the squared norm of a vector.

        Squared norm of a vector associated with the inner-product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector on the tangent space of the hyperbolic space at base point.
        base_point : array-like, shape=[..., dim + 1], optional
            Point in hyperbolic space in extrinsic coordinates.

        Returns
        -------
        sq_norm : array-like, shape=[...,]
            Squared norm of the vector.
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        return sq_norm

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.

        Returns
        -------
        exp : array-like, shape=[..., dim + 1]
            Point in hyperbolic space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        sq_norm_tangent_vec = self.embedding_metric.squared_norm(
            tangent_vec)
        sq_norm_tangent_vec = gs.clip(sq_norm_tangent_vec, 0, math.inf)
        norm_tangent_vec = gs.sqrt(sq_norm_tangent_vec)

        mask_0 = gs.isclose(sq_norm_tangent_vec, 0.)
        mask_0 = gs.to_ndarray(mask_0, to_ndim=1)
        mask_else = ~mask_0
        mask_else = gs.to_ndarray(mask_else, to_ndim=1)
        mask_0_float = gs.cast(mask_0, gs.float32)
        mask_else_float = gs.cast(mask_else, gs.float32)

        coef_1 = gs.zeros_like(norm_tangent_vec)
        coef_2 = gs.zeros_like(norm_tangent_vec)

        coef_1 += mask_0_float * (
            1. + COSH_TAYLOR_COEFFS[2] * norm_tangent_vec ** 2
            + COSH_TAYLOR_COEFFS[4] * norm_tangent_vec ** 4
            + COSH_TAYLOR_COEFFS[6] * norm_tangent_vec ** 6
            + COSH_TAYLOR_COEFFS[8] * norm_tangent_vec ** 8)
        coef_2 += mask_0_float * (
            1. + SINH_TAYLOR_COEFFS[3] * norm_tangent_vec ** 2
            + SINH_TAYLOR_COEFFS[5] * norm_tangent_vec ** 4
            + SINH_TAYLOR_COEFFS[7] * norm_tangent_vec ** 6
            + SINH_TAYLOR_COEFFS[9] * norm_tangent_vec ** 8)
        # This avoids dividing by 0.
        norm_tangent_vec += mask_0_float * 1.0
        coef_1 += mask_else_float * (gs.cosh(norm_tangent_vec))
        coef_2 += mask_else_float * (
            (gs.sinh(norm_tangent_vec) / (norm_tangent_vec)))

        exp = (
            gs.einsum('...,...j->...j', coef_1, base_point)
            + gs.einsum('...,...j->...j', coef_2, tangent_vec))

        hyperbolic_space = Hyperboloid(dim=self.dim)
        exp = hyperbolic_space.regularize(exp)
        return exp

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def log(self, point, base_point):
        """Compute Riemannian logarithm of a point wrt a base point.

        If point_type = 'poincare' then base_point belongs
        to the Poincare ball and point is a vector in the Euclidean
        space of the same dimension as the ball.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.
        base_point : array-like, shape=[..., dim + 1]
            Point in hyperbolic space.

        Returns
        -------
        log : array-like, shape=[..., dim + 1]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        angle = self.dist(base_point, point) / self.scale
        angle = gs.to_ndarray(angle, to_ndim=1)

        mask_0 = gs.isclose(angle, 0.)
        mask_else = ~mask_0

        mask_0_float = gs.cast(mask_0, gs.float32)
        mask_else_float = gs.cast(mask_else, gs.float32)

        coef_1 = gs.zeros_like(angle)
        coef_2 = gs.zeros_like(angle)

        coef_1 += mask_0_float * (
            1. + INV_SINH_TAYLOR_COEFFS[1] * angle ** 2
            + INV_SINH_TAYLOR_COEFFS[3] * angle ** 4
            + INV_SINH_TAYLOR_COEFFS[5] * angle ** 6
            + INV_SINH_TAYLOR_COEFFS[7] * angle ** 8)
        coef_2 += mask_0_float * (
            1. + INV_TANH_TAYLOR_COEFFS[1] * angle ** 2
            + INV_TANH_TAYLOR_COEFFS[3] * angle ** 4
            + INV_TANH_TAYLOR_COEFFS[5] * angle ** 6
            + INV_TANH_TAYLOR_COEFFS[7] * angle ** 8)

        # This avoids dividing by 0.
        angle += mask_0_float * 1.

        coef_1 += mask_else_float * (angle / gs.sinh(angle))
        coef_2 += mask_else_float * (angle / gs.tanh(angle))
        log_term_1 = gs.einsum('...,...j->...j', coef_1, point)
        log_term_2 = - gs.einsum('...,...j->...j', coef_2, base_point)
        log = log_term_1 + log_term_2
        return log

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim + 1]
            First point in hyperbolic space.
        point_b : array-like, shape=[..., dim + 1]
            Second point in hyperbolic space.

        Returns
        -------
        dist : array-like, shape=[...,]
            Geodesic distance between the two points.
        """
        sq_norm_a = self.embedding_metric.squared_norm(point_a)
        sq_norm_b = self.embedding_metric.squared_norm(point_b)
        inner_prod = self.embedding_metric.inner_product(point_a, point_b)

        cosh_angle = - inner_prod / gs.sqrt(sq_norm_a * sq_norm_b)
        cosh_angle = gs.clip(cosh_angle, 1.0, 1e24)

        dist = gs.arccosh(cosh_angle)
        dist *= self.scale
        return dist
