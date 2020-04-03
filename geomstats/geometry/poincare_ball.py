"""The n-dimensional hyperbolic space.

The n-dimensional hyperbolic space embedded with
the hyperboloid representation (embedded in minkowsky space).
"""
import geomstats.backend as gs
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry import hyperbolic

TOLERANCE = 1e-6
EPSILON = 1e-6


class PoincareBall(hyperbolic.Hyperbolic):
    """Class for the n-dimensional hyperbolic space.

    Class for the n-dimensional hyperbolic space
    as embedded in (n+1)-dimensional Minkowski space.

    The point_type variable allows to choose the
    representation of the points as input.

    If point_type is set to 'ball' then points are parametrized
    by their coordinates inside the Poincare Ball n-coordinates.

    Parameters
    ----------
    dimension : int
        Dimension of the hyperbolic space.
    point_type : str, {'extrinsic', 'intrinsic', etc}, optional
        Default coordinates to represent points in hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """
    default_coords_type = 'ball'
    default_point_type = 'vector'

    def __init__(self, dimension, scale=1):
        assert isinstance(dimension, int) and dimension > 0
        super(PoincareBall, self).__init__(
            dimension=dimension,
            embedding_manifold=None,
            scale=scale)
        self.coords_type = PoincareBall.default_coords_type
        self.point_type = PoincareBall.default_point_type
        self.metric =\
            PoincareBallMetric(self.dimension, self.coords_type, self.scale)

    def belongs(self, point, tolerance=TOLERANCE):
        """Test if a point belongs to the hyperbolic space.

        Test if a point belongs to the hyperbolic space based on
        the poincare ball representation, i.e. evaluate if its
        squared norm is lower than 1.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
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
        point = gs.to_ndarray(point, to_ndim=2)
        _, point_dim = point.shape
        if point_dim is not self.dimension + 1:
            if point_dim is self.dimension and self.coords_type == 'intrinsic':
                return gs.array([[True]])
            else:
                return gs.array([[False]])

        sq_norm = self.embedding_metric.squared_norm(point)
        euclidean_sq_norm = gs.linalg.norm(point, axis=-1) ** 2
        euclidean_sq_norm = gs.to_ndarray(euclidean_sq_norm,
                                          to_ndim=2, axis=1)
        diff = gs.abs(sq_norm + 1)
        belongs = diff < tolerance * euclidean_sq_norm
        return belongs


class PoincareBallMetric(RiemannianMetric):
    """Class that defines operations using a hyperbolic metric.

    Parameters
    ----------
    dimension : int
        Dimension of the hyperbolic space.
    point_type : str, {'extrinsic', 'intrinsic', etc}, optional
        Default coordinates to represent points in hyperbolic space.
    scale : int, optional
        Scale of the hyperbolic space, defined as the set of points
        in Minkowski space whose squared norm is equal to -scale.
    """

    default_point_type = 'vector'
    default_coords_type = 'extrinsic'

    def __init__(self, dimension, coords_type='extrinsic', scale=1):
        super(PoincareBallMetric, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.embedding_metric = None

        self.coords_type = coords_type
        self.point_type = PoincareBallMetric.default_point_type

        assert scale > 0, 'The scale should be strictly positive'
        self.scale = scale

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at a base point.
        base_point : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space.

        Returns
        -------
        exp : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        norm_base_point = gs.to_ndarray(
            gs.linalg.norm(base_point, axis=-1), 2, axis=-1)
        norm_base_point = gs.to_ndarray(norm_base_point, to_ndim=2)

        norm_base_point = gs.repeat(
            norm_base_point, base_point.shape[-1], axis=-1)
        den = 1 - norm_base_point**2

        norm_tan = gs.to_ndarray(gs.linalg.norm(
            tangent_vec, axis=-1), 2, axis=-1)
        norm_tan = gs.repeat(norm_tan, base_point.shape[-1], -1)

        lambda_base_point = 1 / den

        direction = tangent_vec / norm_tan

        factor = gs.tanh(lambda_base_point * norm_tan)

        exp = self.mobius_add(base_point, direction * factor)

        zero_tan = gs.isclose((tangent_vec * tangent_vec).sum(axis=-1), 0.)

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
        point : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space.
        base_point : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space.

        Returns
        -------
        log : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        add_base_point = self.mobius_add(-base_point, point)
        norm_add = gs.to_ndarray(gs.linalg.norm(
            add_base_point, axis=-1), 2, -1)
        norm_add = gs.repeat(norm_add, base_point.shape[-1], -1)
        norm_base_point = gs.to_ndarray(gs.linalg.norm(
            base_point, axis=-1), 2, -1)
        norm_base_point = gs.repeat(norm_base_point,
                                    base_point.shape[-1], -1)

        log = (1 - norm_base_point**2) * gs.arctanh(norm_add)\
            * (add_base_point / norm_add)

        mask_0 = gs.isclose(norm_add, 0.)
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
        point_a : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space.
        point_b : array-like, shape=[n_samples, dimension + 1]
            Point in hyperbolic space.

        Returns
        -------
        mobius_add : array-like, shape=[n_samples, 1]
            Result of the Mobius addition.
        """
        norm_point_a = gs.sum(point_a ** 2, axis=-1,
                              keepdims=True)

        # to redefine to use autograd
        norm_point_a = gs.repeat(norm_point_a, point_a.shape[-1], -1)

        norm_point_b = gs.sum(point_b ** 2, axis=-1,
                              keepdims=True)
        norm_point_b = gs.repeat(norm_point_b, point_a.shape[-1], -1)

        sum_prod_a_b = gs.sum(point_a * point_b,
                              axis=-1, keepdims=True)

        sum_prod_a_b = gs.repeat(sum_prod_a_b, point_a.shape[-1], -1)

        add_nominator = ((1 + 2 * sum_prod_a_b + norm_point_b) * point_a +
                         (1 - norm_point_a) * point_b)

        add_denominator = (1 + 2 * sum_prod_a_b + norm_point_a * norm_point_b)

        mobius_add = add_nominator / add_denominator

        return mobius_add


    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dimension + 1]
            First point in hyperbolic space.
        point_b : array-like, shape=[n_samples, dimension + 1]
            Second point in hyperbolic space.

        Returns
        -------
        dist : array-like, shape=[n_samples, 1]
            Geodesic distance between the two points.
        """
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

        Approximate the exponential map of hyperbolic space,
        currently working only with poincare ball.
        .. [1] nickel et.al, "Poincaré Embedding for
         Learning Hierarchical Representation", 2017.


        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension]
            vector in tangent space.
        base_point : array-like, shape=[n_samples, dimension]
            Second point in hyperbolic space.

        Returns
        -------
        point : array-like, shape=[n_samples, dimension]
            Retraction point.
        """
        if self.coords_type == 'ball':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)

            retraction_factor = ((1 - (base_point**2).sum(axis=-1))**2) / 4
            retraction_factor =\
                gs.repeat(gs.expand_dims(retraction_factor, -1),
                          base_point.shape[1],
                          axis=1)
            return base_point - retraction_factor * tangent_vec
        else:
            raise NotImplementedError(
                'Retraction is only implemented for ball and extrinsic')

    def inner_product_matrix(self, base_point=None):
        """Compute the inner product matrix, independent of the base point.

        Parameters
        ----------
        base_point: array-like, shape=[n_samples, dimension]

        Returns
        -------
        inner_prod_mat: array-like, shape=[n_samples, dimension, dimension]
        """

        if(base_point is None):
            base_point = gs.zeros((1, self.dimension))
        dim = base_point.shape[-1]
        n_sample = base_point.shape[0]

        lambda_base =\
            (2 / (1 - gs.sum(base_point * base_point, axis=-1)))**2

        expanded_lambda_base =\
            gs.expand_dims(gs.expand_dims(lambda_base, axis=-1), -1)
        reshaped_lambda_base =\
            gs.repeat(gs.repeat(expanded_lambda_base, dim, axis=-2),
                      dim, axis=-1)

        identity = gs.eye(self.dimension, self.dimension)
        reshaped_identity =\
            gs.repeat(gs.expand_dims(identity, 0), n_sample, axis=0)

        results = reshaped_lambda_base * reshaped_identity
        print("ressss", results)
        return results
