"""Module providing an implementation of MatrixLieAlgebras.

There are two main forms of representation for elements of a MatrixLieAlgebra
implemented here. The first one is as a matrix, as elements of R^(n x n).
The second is by choosing a base and remembering the coefficients of an element
in that base. This base will be provided in child classes
(e.g. SkewSymmetricMatrices).

Lead author: Stefan Heyder.
"""

import abc
import geomstats.backend as gs
from geomstats.spaces.core import LieAlgebra
from geomstats import errors, matrices

from geomstats._bch_coefficients import BCH_COEFFICIENTS


class MatrixLieAlgebra(LieAlgebra, abc.ABC):
    """Class implementing matrix Lie algebra related functions.

    Parameters
    ----------
    dim : int
        Dimension of the Lie algebra as a real vector space.
    n : int
        Amount of rows and columns in the matrix representation of the
        Lie algebra.
    """

    def __init__(self, dim, n, **kwargs):
        super().__init__(shape=(n, n), **kwargs)
        errors.check_integer(dim, "dim")
        errors.check_integer(n, "n")
        self.dim = dim
        self.n = n

    @property
    def default_point_type(self):
        return "matrix"

    def bracket(self, mat_a, mat_b):
        return matrices.bracket(mat_a, mat_b)

    def baker_campbell_hausdorff(self, matrix_a, matrix_b, order=2):
        """Calculate the Baker-Campbell-Hausdorff approximation of given order.

        The implementation is based on [CM2009a]_ with the pre-computed
        constants taken from [CM2009b]_. Our coefficients are truncated to
        enable us to calculate BCH up to order 15.

        This represents Z = log(exp(X)exp(Y)) as an infinite linear combination
        of the form Z = sum z_i e_i where z_i are rational numbers and e_i are
        iterated Lie brackets starting with e_1 = X, e_2 = Y, each e_i is given
        by some i',i'': e_i = [e_i', e_i''].

        Parameters
        ----------
        matrix_a, matrix_b : array-like, shape=[..., n, n]
            Matrices.
        order : int
            The order to which the approximation is calculated. Note that this
            is NOT the same as using only e_i with i < order.
            Optional, default 2.

        References
        ----------
        .. [CM2009a] F. Casas and A. Murua. An efficient algorithm for
            computing the Baker–Campbell–Hausdorff series and some of its
            applications. Journal of Mathematical Physics 50, 2009
        .. [CM2009b] http://www.ehu.eus/ccwmuura/research/bchHall20.dat
        """
        if order > 15:
            raise NotImplementedError("BCH is not implemented for order > 15.")

        number_of_hom_degree = gs.array(
            [2, 1, 2, 3, 6, 9, 18, 30, 56, 99, 186, 335, 630, 1161, 2182]
        )
        n_terms = gs.sum(number_of_hom_degree[:order])

        el = [matrix_a, matrix_b]
        result = matrix_a + matrix_b

        for i in gs.arange(2, n_terms):
            i_p = BCH_COEFFICIENTS[i, 1] - 1
            i_pp = BCH_COEFFICIENTS[i, 2] - 1

            el.append(self.bracket(el[i_p], el[i_pp]))
            result += (
                float(BCH_COEFFICIENTS[i, 3]) / float(BCH_COEFFICIENTS[i, 4]) * el[i]
            )
        return result

    @abc.abstractmethod
    def basis_representation(self, matrix_representation):
        """Compute the coefficients of matrices in the given basis.

        Parameters
        ----------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.

        Returns
        -------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.
        """
        raise NotImplementedError("basis_representation not implemented.")

    def matrix_representation(self, basis_representation):
        """Compute the matrix representation for the given basis coefficients.

        Sums the basis elements according to the coefficients given in
        basis_representation.

        Parameters
        ----------
        basis_representation : array-like, shape=[..., dim]
            Coefficients in the basis.

        Returns
        -------
        matrix_representation : array-like, shape=[..., n, n]
            Matrix.
        """
        if self.basis is None:
            raise NotImplementedError("basis not implemented")

        return gs.einsum("...i,ijk ->...jk", basis_representation, self.basis)

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

        first_term = matrices.mul(inverse_base_point, tangent_vector_b)
        first_term = matrices.mul(tangent_vector_a, first_term)

        second_term = matrices.mul(inverse_base_point, tangent_vector_a)
        second_term = matrices.mul(tangent_vector_b, second_term)

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
        :math:`v` the tangent vector, and :math:`V = g^{-1} v` the associated
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
