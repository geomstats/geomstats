"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded with
the hyperboloid representation (embedded in minkowsky space).
"""

import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry._hyperbolic import _Hyperbolic, HyperbolicMetric
from geomstats.geometry.base import EmbeddedManifold
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.minkowski import MinkowskiMetric


class Hyperboloid(_Hyperbolic, EmbeddedManifold):
    """Class for the n-dimensional hyperboloid space.

    Class for the n-dimensional hyperboloid space as embedded in (
    n+1)-dimensional Minkowski space. For other representations of
    hyperbolic spaces see the `Hyperbolic` class.

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
        minkowski = Minkowski(dim + 1)
        super(Hyperboloid, self).__init__(
            dim=dim, embedding_space=minkowski,
            submersion=minkowski.metric.squared_norm, value=- 1.,
            tangent_submersion=minkowski.metric.inner_product, scale=scale)
        self.coords_type = coords_type
        self.point_type = Hyperboloid.default_point_type
        self.metric =\
            HyperboloidMetric(self.dim, self.coords_type, self.scale)

    def belongs(self, point, atol=gs.atol):
        """Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space in
        its hyperboloid representation.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to be tested.
        atol : float, optional
            Tolerance at which to evaluate how close the squared norm
            is to the reference value.
            Optional, default: backend atol.

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

        return super(Hyperboloid, self).belongs(point, atol)

    def projection(self, point):
        """Project a point in space on the hyperboloid.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in embedding Euclidean space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point projected on the hyperboloid.
        """
        belongs = self.belongs(point)

        # avoid dividing by 0
        factor = gs.where(point[..., 0] == 0., 1., point[..., 0] + gs.atol)

        first_coord = gs.where(belongs, 1., 1. / factor)
        intrinsic = gs.einsum('...,...i->...i', first_coord, point)[..., 1:]
        return self.intrinsic_to_extrinsic_coords(intrinsic)

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

        sq_norm = self.embedding_metric.squared_norm(point)
        if not gs.all(sq_norm):
            raise ValueError('Cannot project a vector of norm 0. in the '
                             'Minkowski space to the hyperboloid')
        real_norm = gs.sqrt(gs.abs(sq_norm))
        projected_point = gs.einsum('...i,...->...i', point, 1. / real_norm)

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

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector.
        base_point : array-like, shape=[..., dim + 1]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        product = self.embedding_metric.inner_product(vector, base_point)
        return gs.isclose(product, 0.)

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
        return _Hyperbolic.change_coordinates_system(
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
            raise NameError("Point that does not belong to the hyperboloid "
                            "found")
        return\
            _Hyperbolic.change_coordinates_system(
                point_extrinsic, 'extrinsic', 'intrinsic')


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

    def metric_matrix(self, base_point=None):
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
        self.embedding_metric.metric_matrix(base_point)

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

        coef_1 = utils.taylor_exp_even_func(
            sq_norm_tangent_vec, utils.cosh_close_0, order=5)
        coef_2 = utils.taylor_exp_even_func(
            sq_norm_tangent_vec, utils.sinch_close_0, order=5)

        exp = (
            gs.einsum('...,...j->...j', coef_1, base_point)
            + gs.einsum('...,...j->...j', coef_2, tangent_vec))

        exp = Hyperboloid(dim=self.dim).regularize(exp)
        return exp

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

        coef_1_ = utils.taylor_exp_even_func(
            angle ** 2, utils.inv_sinch_close_0, order=4)
        coef_2_ = utils.taylor_exp_even_func(
            angle ** 2, utils.inv_tanh_close_0, order=4)

        log_term_1 = gs.einsum('...,...j->...j', coef_1_, point)
        log_term_2 = - gs.einsum('...,...j->...j', coef_2_, base_point)
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

    def parallel_transport(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic defined by exp_(base_point)(tangent_vec_b).

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., dim + 1]
            Transported tangent vector at exp_(base_point)(tangent_vec_b).
        """
        theta = self.embedding_metric.norm(tangent_vec_b)
        eps = gs.where(theta == 0., 1., theta)
        normalized_b = gs.einsum('...,...i->...i', 1 / eps, tangent_vec_b)
        pb = self.embedding_metric.inner_product(tangent_vec_a, normalized_b)
        p_orth = tangent_vec_a - gs.einsum('...,...i->...i', pb, normalized_b)
        transported = \
            gs.einsum('...,...i->...i', gs.sinh(theta) * pb, base_point)\
            + gs.einsum('...,...i->...i', gs.cosh(theta) * pb, normalized_b)\
            + p_orth
        return transported
