"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded with
the hyperboloid representation (embedded in minkowsky space).
"""
import geomstats.backend as gs
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-6


class PoincareBall(Hyperbolic):
    """Class for the n-dimensional hyperbolic space.

    Class for the n-dimensional hyperbolic space
    as embedded in the Poincaré ball model.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    default_coords_type = 'ball'
    default_point_type = 'vector'

    def __init__(self, dim, scale=1):
        super(PoincareBall, self).__init__(
            dim=dim,
            scale=scale)
        self.coords_type = PoincareBall.default_coords_type
        self.point_type = PoincareBall.default_point_type
        self.metric =\
            PoincareBallMetric(self.dim, self.scale)

    def belongs(self, point, tolerance=TOLERANCE):
        """Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space based on
        the poincare ball representation.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim]
            Point to be tested.
        tolerance : float, optional
            Tolerance at which to evaluate how close the squared norm
            is to the reference value.

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            Array of booleans indicating whether the corresponding points
            belong to the hyperbolic space.
        """
        return gs.sum(point**2, axis=-1) < (1 - tolerance)


class PoincareBallMetric(RiemannianMetric):
    """Class that defines operations using a Poincare ball.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    default_point_type = 'vector'
    default_coords_type = 'ball'

    def __init__(self, dim, scale=1):
        super(PoincareBallMetric, self).__init__(
            dim=dim,
            signature=(dim, 0, 0))
        self.coords_type = PoincareBall.default_coords_type
        self.point_type = PoincareBall.default_point_type
        self.scale = scale

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dim]
            Tangent vector at a base point.
        base_point : array-like, shape=[n_samples, dim]
            Point in hyperbolic space.

        Returns
        -------
        exp : array-like, shape=[n_samples, dim]
            Point in hyperbolic space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        norm_base_point =\
            gs.expand_dims(gs.linalg.norm(base_point, axis=-1),
                           axis=-1)

        den = 1 - norm_base_point**2

        norm_tan =\
            gs.expand_dims(gs.linalg.norm(tangent_vec, axis=-1),
                           axis=-1)

        lambda_base_point = 1 / den

        zero_tan =\
            gs.isclose(gs.sum(tangent_vec * tangent_vec, axis=-1), 0.)

        if gs.any(zero_tan):
            if norm_tan[zero_tan].shape[0] != 0:
                norm_tan[zero_tan] = EPSILON

        direction = gs.einsum('...i,...k->...i', tangent_vec, 1 / norm_tan)

        factor = gs.tanh(lambda_base_point * norm_tan)

        exp = self.mobius_add(base_point, direction * factor)

        zero_tan =\
            gs.isclose(gs.sum(tangent_vec * tangent_vec, axis=-1), 0.)

        if gs.any(zero_tan):
            if exp[zero_tan].shape[0] != 0:
                exp[zero_tan] = base_point[zero_tan]

        return exp

    def log(self, point, base_point):
        """Compute Riemannian logarithm of a point wrt a base point.

        If point_type = 'poincare' then base_point belongs
        to the Poincare ball and point is a vector in the Euclidean
        space of the same dimension as the ball.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim]
            Point in hyperbolic space.
        base_point : array-like, shape=[n_samples, dim]
            Point in hyperbolic space.

        Returns
        -------
        log : array-like, shape=[n_samples, dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        point = gs.to_ndarray(point, to_ndim=2)

        add_base_point = self.mobius_add(-base_point, point)
        norm_add =\
            gs.expand_dims(gs.linalg.norm(
                           add_base_point, axis=-1), axis=-1)

        norm_base_point =\
            gs.expand_dims(gs.linalg.norm(
                           base_point, axis=-1), axis=-1)

        log = (1 - norm_base_point**2) * gs.arctanh(norm_add)
        log = gs.einsum('...i,...j->...j', log, (add_base_point / norm_add))

        mask_0 = gs.isclose(gs.squeeze(norm_add, axis=-1), 0.)
        if gs.any(mask_0):
            log[mask_0] = 0

        return log

    def mobius_add(self, point_a, point_b):
        r"""Compute the Mobius addition of two points.

        Mobius addition operation that is a necessary operation
        to compute the log and exp using the 'ball' representation.

        .. math::

            a\oplus b=\frac{(1+2\langle a,b\rangle + ||b||^2)a+
            (1-||a||^2)b}{1+2\langle a,b\rangle + ||a||^2||b||^2}

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dim]
            Point in hyperbolic space.
        point_b : array-like, shape=[n_samples, dim]
            Point in hyperbolic space.

        Returns
        -------
        mobius_add : array-like, shape=[n_samples, 1]
            Result of the Mobius addition.
        """
        point_a = gs.to_ndarray(point_a, to_ndim=2)
        point_b = gs.to_ndarray(point_b, to_ndim=2)

        ball_manifold = PoincareBall(self.dim, scale=self.scale)
        point_a_belong = ball_manifold.belongs(point_a)
        point_b_belong = ball_manifold.belongs(point_b)

        if(not gs.all(point_a_belong) or not gs.all(point_b_belong)):
            raise NameError("Points do not belong to the Poincare ball")

        norm_point_a = gs.sum(point_a ** 2, axis=-1,
                              keepdims=True)

        norm_point_b = gs.sum(point_b ** 2, axis=-1,
                              keepdims=True)

        sum_prod_a_b = gs.einsum('...i,...i->...', point_a, point_b)
        sum_prod_a_b = gs.expand_dims(sum_prod_a_b, axis=-1)

        add_num_1 = 1 + 2 * sum_prod_a_b + norm_point_b
        add_num_1 = gs.einsum('...i,...k->...k', add_num_1, point_a)
        add_num_2 = gs.einsum('...i,...k->...k', (1 - norm_point_a), point_b)
        add_nominator = add_num_1 + add_num_2

        add_denominator = (1 + 2 * sum_prod_a_b + norm_point_a * norm_point_b)

        mobius_add =\
            gs.einsum('...i,...k->...i', add_nominator, 1 / add_denominator)

        return mobius_add

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dim]
            First point in hyperbolic space.
        point_b : array-like, shape=[n_samples, dim]
            Second point in hyperbolic space.

        Returns
        -------
        dist : array-like, shape=[n_samples, 1]
            Geodesic distance between the two points.
        """
        point_a = gs.to_ndarray(point_a, to_ndim=2)
        point_b = gs.to_ndarray(point_b, to_ndim=2)

        point_a_norm = gs.clip(gs.sum(point_a ** 2, -1), 0., 1 - EPSILON)
        point_b_norm = gs.clip(gs.sum(point_b ** 2, -1), 0., 1 - EPSILON)

        diff_norm = gs.sum((point_a - point_b) ** 2, -1)
        norm_function = 1 + 2 * \
            diff_norm / ((1 - point_a_norm) * (1 - point_b_norm))

        dist = gs.log(norm_function + gs.sqrt(norm_function ** 2 - 1))
        dist = gs.to_ndarray(dist, to_ndim=1)
        dist = gs.to_ndarray(dist, to_ndim=2, axis=1)
        dist *= self.scale
        return dist

    def retraction(self, tangent_vec, base_point):
        """Poincaré ball model retraction.

        Approximate the exponential map of hyperbolic space
        .. [1] nickel et.al, "Poincaré Embedding for
         Learning Hierarchical Representation", 2017.


        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dim]
            vector in tangent space.
        base_point : array-like, shape=[n_samples, dim]
            Second point in hyperbolic space.

        Returns
        -------
        point : array-like, shape=[n_samples, dim]
            Retraction point.
        """
        ball_manifold = PoincareBall(self.dim, scale=self.scale)
        base_point_belong = ball_manifold.belongs(base_point)

        if not gs.all(base_point_belong):
            raise NameError("Points do not belong to the Poincare ball")

        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        retraction_factor =\
            ((1 - gs.sum(base_point**2, axis=-1, keepdims=True))**2) / 4

        return base_point\
            - gs.einsum('...i,...j->...j', retraction_factor, tangent_vec)

    def inner_product_matrix(self, base_point=None):
        """Compute the inner product matrix.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dim]

        Returns
        -------
        inner_prod_mat: array-like, shape=[n_samples, dim, dim]
        """
        if base_point is None:
            base_point = gs.zeros((1, self.dim))

        lambda_base =\
            (2 / (1 - gs.sum(base_point * base_point, axis=-1)))**2
        identity = gs.eye(self.dim, self.dim)

        return gs.einsum('i,jk->ijk', lambda_base, identity)
