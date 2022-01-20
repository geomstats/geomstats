"""Lie groups.

Lead author: Nina Miolane.
"""

import abc

import geomstats.backend as gs
import geomstats.errors as errors
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices

ATOL = 1e-6


class MatrixLieGroup(Manifold, abc.ABC):
    """Class for matrix Lie groups."""

    def __init__(self, dim, n, lie_algebra=None, **kwargs):
        super(MatrixLieGroup, self).__init__(dim=dim, shape=(n, n), **kwargs)
        self.lie_algebra = lie_algebra
        self.n = n
        self.left_canonical_metric = InvariantMetric(
            group=self, metric_mat_at_identity=gs.eye(self.dim), left_or_right="left"
        )

        self.right_canonical_metric = InvariantMetric(
            group=self, metric_mat_at_identity=gs.eye(self.dim), left_or_right="right"
        )

    @property
    def identity(self):
        """Matrix identity."""
        return gs.eye(self.n)

    @staticmethod
    def compose(point_a, point_b):
        """Perform function composition corresponding to the Lie group.

        Multiply the elements `point_a` and `point_b`.

        Parameters
        ----------
        point_a : array-like, shape=[..., {dim, [n, n]}]
            Left factor in the product.
        point_b : array-like, shape=[..., {dim, [n, n]}]
            Right factor in the product.

        Returns
        -------
        composed : array-like, shape=[..., {dim, [n, n]}]
            Product of point_a and point_b along the first dimension.
        """
        return Matrices.mul(point_a, point_b)

    @classmethod
    def inverse(cls, point):
        """Compute the inverse law of the Lie group.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point to be inverted.

        Returns
        -------
        inverse : array-like, shape=[..., {dim, [n, n]}]
            Inverted point.
        """
        return gs.linalg.inv(point)

    def tangent_translation_map(self, point, left_or_right="left", inverse=False):
        r"""Compute the push-forward map by the left/right translation.

        Compute the push-forward map, of the left/right translation by the
        point. It corresponds to the tangent map, or differential of the
        group multiplication by the point or its inverse. For groups with a
        vector representation, it is only implemented at identity, but it can
        be used at other points by passing `inverse=True`. This method wraps
        the jacobian translation which actually computes the matrix
        representation of the map.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]]
            Point.
        left_or_right : str, {'left', 'right'}
            Whether to calculate the differential of left or right
            translations.
            Optional, default: 'left'
        inverse : bool,
            Whether to inverse the jacobian matrix. If True, the push forward
            by the translation by the inverse of point is returned.
            Optional, default: False.

        Returns
        -------
        tangent_map : callable
            Tangent map of the left/right translation by point. It can be
            applied to tangent vectors.
        """
        errors.check_parameter_accepted_values(
            left_or_right, "left_or_right", ["left", "right"]
        )
        if inverse:
            point = self.inverse(point)
        if left_or_right == "left":
            return lambda tangent_vec: self.compose(point, tangent_vec)
        return lambda tangent_vec: self.compose(tangent_vec, point)

    def lie_bracket(self, tangent_vector_a, tangent_vector_b, base_point=None):
        """Compute the lie bracket of two tangent vectors.

        For matrix Lie groups with tangent vectors A,B at the same base point P
        this is given by (translate to identity, compute commutator, go back)
        :math:`[A,B] = A_P^{-1}B - B_P^{-1}A`

        Parameters
        ----------
        tangent_vector_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vector_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        bracket : array-like, shape=[..., n, n]
            Lie bracket.
        """
        if base_point is None:
            base_point = self.identity
        inverse_base_point = self.inverse(base_point)

        first_term = Matrices.mul(inverse_base_point, tangent_vector_b)
        first_term = Matrices.mul(tangent_vector_a, first_term)

        second_term = Matrices.mul(inverse_base_point, tangent_vector_a)
        second_term = Matrices.mul(tangent_vector_b, second_term)

        return first_term - second_term

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim_embedding]
            Vector.
        base_point : array-like, shape=[..., dim_embedding]
            Point in the Lie group.
            Optional. default: identity.
        atol : float
            Precision at which to evaluate if the rotation part is
            skew-symmetric.
            Optional. default: 1e-6

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        if base_point is None:
            tangent_vec_at_id = vector
        else:
            tangent_vec_at_id = self.compose(self.inverse(base_point), vector)
        is_tangent = self.lie_algebra.belongs(tangent_vec_at_id, atol)
        return is_tangent

    def to_tangent(self, vector, base_point=None):
        """Project a vector onto the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector to project. Its shape must match the shape of base_point.
        base_point : array-like, shape=[..., {dim, [n, n]}], optional
            Point of the group.
            Optional, default: identity.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        """
        if base_point is None:
            return self.lie_algebra.projection(vector)
        tangent_vec_at_id = self.compose(self.inverse(base_point), vector)
        regularized = self.lie_algebra.projection(tangent_vec_at_id)
        return self.compose(base_point, regularized)

    @classmethod
    def exp(cls, tangent_vec, base_point=None):
        r"""
        Exponentiate a left-invariant vector field from a base point.

        The vector input is not an element of the Lie algebra, but of the
        tangent space at base_point: if :math:`g` denotes `base_point`,
        :math:`v` the tangent vector, and :math:'V = g^{-1} v' the associated
        Lie algebra vector, then

        .. math::

            \exp(v, g) = mul(g, \exp(V))

        Therefore, the Lie exponential is obtained when base_point is None, or
        the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, defaults to identity if None.

        Returns
        -------
        point : array-like, shape=[..., n, n]
            Left multiplication of `exp(algebra_mat)` with `base_point`.
        """
        expm = gs.linalg.expm
        if base_point is None:
            return expm(tangent_vec)
        lie_algebra_vec = cls.compose(cls.inverse(base_point), tangent_vec)
        return cls.compose(base_point, cls.exp(lie_algebra_vec))

    @classmethod
    def log(cls, point, base_point=None):
        r"""
        Compute a left-invariant vector field bringing base_point to point.

        The output is a vector of the tangent space at base_point, so not a Lie
        algebra element if it is not the identity.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, defaults to identity if None.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Matrix such that `exp(tangent_vec, base_point) = point`.

        Notes
        -----
        Denoting `point` by :math:`g` and `base_point` by :math:`h`,
        the output satisfies:

        .. math::

            g = \exp(\log(g, h), h)
        """
        logm = gs.linalg.logm
        if base_point is None:
            return logm(point)
        lie_algebra_vec = logm(cls.compose(cls.inverse(base_point), point))
        return cls.compose(base_point, lie_algebra_vec)


class LieGroup(Manifold, abc.ABC):
    """Class for Lie groups.

    In this class, point_type ('vector' or 'matrix') will be used to describe
    the format of the points on the Lie group.
    If point_type is 'vector', the format of the inputs is
    [..., dimension], where dimension is the dimension of the Lie group.
    If point_type is 'matrix' the format of the inputs is
    [..., n, n] where n is the parameter of GL(n) e.g. the amount of rows
    and columns of the matrix.

    Parameters
    ----------
    dim : int
        Dimension of the Lie group.
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    lie_algebra : MatrixLieAlgebra
        Lie algebra for matrix groups.
        Optional, default: None.

    Attributes
    ----------
    lie_algebra : MatrixLieAlgebra or None
        Tangent space at the identity.
    left_canonical_metric : InvariantMetric
        The left invariant metric that corresponds to the Euclidean inner
        product at the identity.
    right_canonical_metric : InvariantMetric
        The right invariant metric that corresponds to the Euclidean inner
        product at the identity.
    """

    def __init__(
        self, dim, shape, default_point_type="vector", lie_algebra=None, **kwargs
    ):
        super(LieGroup, self).__init__(
            dim=dim, shape=shape, default_point_type=default_point_type, **kwargs
        )

        self.lie_algebra = lie_algebra
        self.left_canonical_metric = InvariantMetric(
            group=self, metric_mat_at_identity=gs.eye(self.dim), left_or_right="left"
        )

        self.right_canonical_metric = InvariantMetric(
            group=self, metric_mat_at_identity=gs.eye(self.dim), left_or_right="right"
        )

        self.metric = self.left_canonical_metric
        self.metrics = []

    def get_identity(self, point_type=None):
        """Get the identity of the group.

        Parameters
        ----------
        point_type : str, {'matrix', 'vector'}
            Point type.
            Optional, default: None.

        Returns
        -------
        identity : array-like, shape={[dim], [n, n]}
            Identity of the Lie group.
        """
        raise NotImplementedError("The Lie group identity is not implemented.")

    identity = property(get_identity)

    def compose(self, point_a, point_b):
        """Perform function composition corresponding to the Lie group.

        Multiply the elements `point_a` and `point_b`.

        Parameters
        ----------
        point_a : array-like, shape=[..., {dim, [n, n]}]
            Left factor in the product.
        point_b : array-like, shape=[..., {dim, [n, n]}]
            Right factor in the product.

        Returns
        -------
        composed : array-like, shape=[..., {dim, [n, n]}]
            Product of point_a and point_b along the first dimension.
        """
        raise NotImplementedError("The Lie group composition is not implemented.")

    @classmethod
    def inverse(cls, point):
        """Compute the inverse law of the Lie group.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point to be inverted.

        Returns
        -------
        inverse : array-like, shape=[..., {dim, [n, n]}]
            Inverted point.
        """
        raise NotImplementedError("The Lie group inverse is not implemented.")

    def jacobian_translation(self, point, left_or_right="left"):
        """Compute the Jacobian of left/right translation by a point.

        Compute the Jacobian matrix of the left translation by the point.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]]
            Point.
        left_or_right : str, {'left', 'right'}
            Indicate whether to calculate the differential of left or right
            translations.
            Optional, default: 'left'.

        Returns
        -------
        jacobian : array-like, shape=[..., dim, dim]
            Jacobian of the left/right translation by point.
        """
        raise NotImplementedError(
            "The jacobian of the Lie group translation is not implemented."
        )

    def tangent_translation_map(self, point, left_or_right="left", inverse=False):
        r"""Compute the push-forward map by the left/right translation.

        Compute the push-forward map, of the left/right translation by the
        point. It corresponds to the tangent map, or differential of the
        group multiplication by the point or its inverse. For groups with a
        vector representation, it is only implemented at identity, but it can
        be used at other points by passing `inverse=True`. This method wraps
        the jacobian translation which actually computes the matrix
        representation of the map.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]]
            Point.
        left_or_right : str, {'left', 'right'}
            Whether to calculate the differential of left or right
            translations.
            Optional, default: 'left'
        inverse : bool,
            Whether to inverse the jacobian matrix. If True, the push forward
            by the translation by the inverse of point is returned.
            Optional, default: False.

        Returns
        -------
        tangent_map : callable
            Tangent map of the left/right translation by point. It can be
            applied to tangent vectors.
        """
        errors.check_parameter_accepted_values(
            left_or_right, "left_or_right", ["left", "right"]
        )
        if self.default_point_type == "matrix":
            if inverse:
                point = self.inverse(point)
            if left_or_right == "left":
                return lambda tangent_vec: Matrices.mul(point, tangent_vec)
            return lambda tangent_vec: Matrices.mul(tangent_vec, point)

        jacobian = self.jacobian_translation(point, left_or_right)
        if inverse:
            jacobian = gs.linalg.inv(jacobian)
        return lambda tangent_vec: gs.einsum("...ij,...j->...i", jacobian, tangent_vec)

    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of tangent vector from the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at base point.

        Returns
        -------
        point : array-like, shape=[..., {dim, [n, n]}]
            Group exponential.
        """
        raise NotImplementedError(
            "The group exponential from the identity is not implemented."
        )

    def exp_not_from_identity(self, tangent_vec, base_point):
        """Calculate the group exponential at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at base point.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., {dim, [n, n]}]
            Group exponential.
        """
        if self.default_point_type == "vector":
            tangent_translation = self.tangent_translation_map(
                point=base_point, left_or_right="left", inverse=True
            )

            tangent_vec_at_id = tangent_translation(tangent_vec)
            exp_from_identity = self.exp_from_identity(tangent_vec=tangent_vec_at_id)
            exp = self.compose(base_point, exp_from_identity)
            exp = self.regularize(exp)
            return exp

        lie_vec = self.compose(self.inverse(base_point), tangent_vec)
        return self.compose(base_point, self.exp_from_identity(lie_vec))

    def exp(self, tangent_vec, base_point=None):
        """Compute the group exponential at `base_point` of `tangent_vec`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Tangent vector at base point.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Base point.
            Optional, default: self.identity

        Returns
        -------
        result : array-like, shape=[..., {dim, [n, n]}]
            Group exponential.
        """
        identity = self.get_identity()

        if base_point is None:
            base_point = identity
        base_point = self.regularize(base_point)

        if gs.allclose(base_point, identity):
            result = self.exp_from_identity(tangent_vec)
        else:
            result = self.exp_not_from_identity(tangent_vec, base_point)
        return result

    def log_from_identity(self, point):
        """Compute the group logarithm of `point` from the identity.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Group logarithm.
        """
        raise NotImplementedError(
            "The group logarithm from the identity is not implemented."
        )

    def log_not_from_identity(self, point, base_point):
        """Compute the group logarithm of `point` from `base_point`.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Group logarithm.
        """
        if self.default_point_type == "vector":
            tangent_translation = self.tangent_translation_map(
                point=base_point, left_or_right="left"
            )
            point_near_id = self.compose(self.inverse(base_point), point)
            log_from_id = self.log_from_identity(point=point_near_id)
            log = tangent_translation(log_from_id)
            return log

        lie_point = self.compose(self.inverse(base_point), point)
        return self.compose(base_point, self.log_from_identity(lie_point))

    def log(self, point, base_point=None):
        """Compute the group logarithm of `point` relative to `base_point`.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
            Point.
        base_point : array-like, shape=[..., {dim, [n, n]}]
            Base point.
            Optional, defaults to identity if None.

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            Group logarithm.
        """
        # TODO (ninamiolane): Build a standalone decorator that *only*
        # deals with point_type None and base_point None
        identity = self.get_identity(point_type=self.default_point_type)
        if base_point is None:
            base_point = identity

        point = self.regularize(point)
        base_point = self.regularize(base_point)

        if gs.allclose(base_point, identity):
            result = self.log_from_identity(point)
        else:
            result = self.log_not_from_identity(point, base_point)
        return result

    def add_metric(self, metric):
        """Add a metric to the instance's list of metrics.

        Parameters
        ----------
        metric : RiemannianMetric
            Metric to add.
        """
        self.metrics.append(metric)

    def lie_bracket(self, tangent_vector_a, tangent_vector_b, base_point=None):
        """Compute the lie bracket of two tangent vectors.

        For matrix Lie groups with tangent vectors A,B at the same base point P
        this is given by (translate to identity, compute commutator, go back)
        :math:`[A,B] = A_P^{-1}B - B_P^{-1}A`

        Parameters
        ----------
        tangent_vector_a : array-like, shape=[..., n, n]
            Tangent vector at base point.
        tangent_vector_b : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        bracket : array-like, shape=[..., n, n]
            Lie bracket.
        """
        if base_point is None:
            base_point = self.get_identity(point_type=self.default_point_type)
        inverse_base_point = self.inverse(base_point)

        first_term = Matrices.mul(inverse_base_point, tangent_vector_b)
        first_term = Matrices.mul(tangent_vector_a, first_term)

        second_term = Matrices.mul(inverse_base_point, tangent_vector_a)
        second_term = Matrices.mul(tangent_vector_b, second_term)

        return first_term - second_term

    def is_tangent(self, vector, base_point=None, atol=ATOL):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim_embedding]
            Vector.
        base_point : array-like, shape=[..., dim_embedding]
            Point in the Lie group.
            Optional. default: identity.
        atol : float
            Precision at which to evaluate if the rotation part is
            skew-symmetric.
            Optional. default: 1e-6

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        if base_point is None:
            tangent_vec_at_id = vector
        else:
            tangent_vec_at_id = self.compose(self.inverse(base_point), vector)
        is_tangent = self.lie_algebra.belongs(tangent_vec_at_id, atol)
        return is_tangent

    def to_tangent(self, vector, base_point=None):
        """Project a vector onto the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector to project. Its shape must match the shape of base_point.
        base_point : array-like, shape=[..., {dim, [n, n]}], optional
            Point of the group.
            Optional, default: identity.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        """
        if base_point is None:
            return self.lie_algebra.projection(vector)
        tangent_vec_at_id = self.compose(self.inverse(base_point), vector)
        regularized = self.lie_algebra.projection(tangent_vec_at_id)
        return self.compose(base_point, regularized)
