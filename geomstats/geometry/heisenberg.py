"""The 3D Heisenberg group.

Lead author: Morten Pedersen.
"""

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.lie_group import LieGroup


def _scalarmatmul(scalar, mat):
    """Vectorized scalar matrix multiplication.

    Parameters
    ----------
    scalar : array-like, shape=[...]
    mat : array-like, shape=[n, m]
    """
    return gs.einsum("...,ij->...ij", scalar, mat)


class HeisenbergVectorsToMatrices(Diffeo):
    r"""Diffeomorphism from the vector to the matrix representation.

    The 3D Heisenberg group can also be represented as the group of 3x3 upper
    triangular matrices with unit diagonal. In exponential coordinates, the
    diffeomorphism is the group exponential (composed with the hat map)

    .. math::

        (x, y, z) \mapsto \begin{pmatrix}
        1 & x & z + \frac{1}{2} x y \\
        0 & 1 & y \\
        0 & 0 & 1
        \end{pmatrix}

    It is also a group isomorphism.
    """

    @staticmethod
    def _matrix_from_entries(entry_01, entry_12, entry_02, diag):
        """Build a 3x3 upper triangular matrix from its entries.

        Parameters
        ----------
        entry_01 : array-like, shape=[...]
        entry_12 : array-like, shape=[...]
        entry_02 : array-like, shape=[...]
        diag : float
            Value of the diagonal entries.

        Returns
        -------
        matrix : array-like, shape=[..., 3, 3]
        """
        zeros = gs.zeros_like(entry_02)
        diag = zeros + diag
        entry_01 = gs.broadcast_to(entry_01, entry_02.shape)
        entry_12 = gs.broadcast_to(entry_12, entry_02.shape)
        entries = gs.stack(
            [diag, entry_01, entry_02, zeros, diag, entry_12, zeros, zeros, diag],
            axis=-1,
        )
        return gs.reshape(entries, entry_02.shape + (3, 3))

    def __call__(self, base_point):
        """Compute the upper triangular matrix representation of a point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 3]
            Point in the vector representation.

        Returns
        -------
        image_point : array-like, shape=[..., 3, 3]
            Upper triangular matrix with unit diagonal.
        """
        x, y, z = base_point[..., 0], base_point[..., 1], base_point[..., 2]
        return self._matrix_from_entries(x, y, z + x * y / 2, 1.0)

    def inverse(self, image_point):
        """Compute the vector representation of an upper triangular matrix.

        Parameters
        ----------
        image_point : array-like, shape=[..., 3, 3]
            Upper triangular matrix with unit diagonal.

        Returns
        -------
        base_point : array-like, shape=[..., 3]
            Point in the vector representation.
        """
        a, b, c = image_point[..., 0, 1], image_point[..., 1, 2], image_point[..., 0, 2]
        return gs.stack([a, b, c - a * b / 2], axis=-1)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Tangent diffeomorphism at base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 3]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 3]
            Base point.
        image_point : array-like, shape=[..., 3, 3]
            Image point.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 3, 3]
            Image tangent vector at image of the base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)

        x, y = base_point[..., 0], base_point[..., 1]
        dx, dy, dz = tangent_vec[..., 0], tangent_vec[..., 1], tangent_vec[..., 2]
        return self._matrix_from_entries(dx, dy, dz + (x * dy + y * dx) / 2, 0.0)

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Inverse tangent diffeomorphism at image point.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 3, 3]
            Image tangent vector at image point.
        image_point : array-like, shape=[..., 3, 3]
            Image point.
        base_point : array-like, shape=[..., 3]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 3]
            Tangent vector at base point.
        """
        if image_point is None:
            image_point = self(base_point)

        a, b = image_point[..., 0, 1], image_point[..., 1, 2]
        da, db, dc = (
            image_tangent_vec[..., 0, 1],
            image_tangent_vec[..., 1, 2],
            image_tangent_vec[..., 0, 2],
        )
        dz = dc - (a * db + b * da) / 2
        return gs.stack(
            [gs.broadcast_to(da, dz.shape), gs.broadcast_to(db, dz.shape), dz],
            axis=-1,
        )


class HeisenbergVectors(LieGroup, VectorSpace):
    """Class for the 3D Heisenberg group in the vector representation.

    The 3D Heisenberg group represented as R^3. It is a step-2 Carnot Lie
    group. It can be equipped with a natural sub-Riemannian structure, and it
    is a fundamental example in sub-Riemannian geometry.

    Parameters
    ----------
    No parameters

    References
    ----------
    https://en.wikipedia.org/wiki/Heisenberg_group
    """

    def __init__(self, equip=True):
        super().__init__(dim=3, shape=(3,), lie_algebra=Euclidean(3), equip=equip)

    def _create_basis(self):
        """Create the canonical basis."""
        return gs.eye(3)

    @property
    def identity(self):
        """Identity of the 3D Heisenberg group.

        Returns
        -------
        _ : array-like, shape=[3,]
        """
        return gs.zeros(self.dim)

    def compose(self, point_a, point_b):
        """Compute the group product of elements `point_a` and `point_b`.

        Parameters
        ----------
        point_a : array-like, shape=[..., 3]
            Left factor in the product.
        point_b : array-like, shape=[..., 3]
            Right factor in the product.

        Returns
        -------
        point_ab : array-like, shape=[..., 3]
            Product of point_a and point_b along the first dimension.
        """
        point_ab = point_a + point_b
        point_ab_additional_term = gs.array(
            1
            / 2
            * (point_a[..., 0] * point_b[..., 1] - point_a[..., 1] * point_b[..., 0])
        )

        point_ab = point_ab + gs.concatenate(
            [gs.zeros_like(point_ab[..., :2]), point_ab_additional_term[..., None]],
            axis=-1,
        )

        return point_ab

    def inverse(self, point):
        """Compute the group inverse of point.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.

        Returns
        -------
        _ : array-like, shape=[..., 3]
            Inverse.
        """
        return -point

    def jacobian_translation(self, point, left=True):
        """Compute the Jacobian matrix of left/right translation by a point.

        This calculates the differential of the left translation L_(point)
        evaluated at `point`. Note that it only depends on the point we are
        left-translating by, not on the point where the differential is
        evaluated.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.
        left : bool
            Indicate whether to calculate the differential of left or right
            translations.
            Optional, default: True.

        Returns
        -------
        _ : array-like, shape=[..., 3, 3]
            Jacobian of the left/right translation by point.
        """
        e31 = gs.array_from_sparse([(2, 0)], [1.0], (3, 3))
        e32 = gs.array_from_sparse([(2, 1)], [1.0], (3, 3))

        if left:
            return (
                gs.eye(3)
                + _scalarmatmul(-point[..., 1] / 2, e31)
                + _scalarmatmul(point[..., 0] / 2, e32)
            )

        return (
            gs.eye(3)
            + _scalarmatmul(point[..., 1] / 2, e31)
            + _scalarmatmul(-point[..., 0] / 2, e32)
        )

    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of the tangent vector at the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 3]
            Tangent vector at base point.

        Returns
        -------
        _ : array-like, shape=[..., 3]
            Point.
        """
        return gs.copy(tangent_vec)

    def log_from_identity(self, point):
        """Compute the group logarithm of the point at the identity.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.

        Returns
        -------
        _ : array-like, shape=[..., 3]
            Group logarithm.
        """
        return gs.copy(point)

    def lie_bracket(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the lie bracket of two tangent vectors."""
        raise NotImplementedError("The lie bracket is not implemented.")
