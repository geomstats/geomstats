"""The 3D Heisenberg group.

Lead author: Morten Pedersen.
"""

import geomstats.backend as gs
from geomstats.geometry.base import VectorSpace
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.symmetric_matrices import SymmetricMatrices


class HeisenbergVectors(LieGroup, VectorSpace):
    """Class for the 3D Heisenberg group in the vector representation.

    The 3D Heisenberg group represented as R^3. It is a step-2 Carnot Lie
    group. It can be equipped with a natural sub-Riemannian structure, and it is
    it a fundamental example in sub-Riemannian geometry.

    Parameters
    ----------
    No parameters

    Reference
    ---------
    https://en.wikipedia.org/wiki/Heisenberg_group
    """

    def __init__(self, **kwargs):
        super(HeisenbergVectors, self).__init__(
            dim=3, shape=(3,), lie_algebra=Euclidean(3)
        )

    def _create_basis(self):
        """Create the canonical basis."""
        return gs.eye(3)

    def get_identity(self, point_type="vector"):
        """Get the identity of the 3D Heisenberg group.

        Parameters
        ----------
        point_type : str, {'vector', 'matrix'}
            Point_type of the returned value. Unused here.

        Returns
        -------
        _ : array-like, shape=[3,]
        """
        return gs.zeros(self.dim)

    identity = property(get_identity)

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

    def jacobian_translation(self, point, left_or_right="left"):
        """Compute the Jacobian matrix of left/right translation by a point.

        This calculates the differential of the left translation L_(point)
        evaluated at 'point'. Note that it only depends on the point we are
        left-translating by, not on the point where the differential is evaluated.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.
        left_or_right : str, {'left', 'right'}
            Indicate whether to calculate the differential of left or right
            translations.
            Optional, default: 'left'.

        Returns
        -------
        _ : array-like, shape=[..., 3, 3]
            Jacobian of the left/right translation by point.
        """
        e31 = gs.array_from_sparse([(2, 0)], [1.0], (3, 3))
        e32 = gs.array_from_sparse([(2, 1)], [1.0], (3, 3))

        if left_or_right == "left":
            return (
                gs.eye(3)
                + gs.einsum("..., ij -> ...ij", -point[..., 1] / 2, e31)
                + gs.einsum("..., ij -> ...ij", point[..., 0] / 2, e32)
            )

        return (
            gs.eye(3)
            + gs.einsum("..., ij -> ...ij", point[..., 1] / 2, e31)
            + gs.einsum("..., ij -> ...ij", -point[..., 0] / 2, e32)
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
        return tangent_vec

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
        return point

    @staticmethod
    def upper_triangular_matrix_from_vector(point):
        """Compute the upper triangular matrix representation of the vector.

        The 3D Heisenberg group can also be represented as 3x3 upper triangular
        matrices. This function computes this representation of the vector
        'point'.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point in the vector-represention.

        Returns
        -------
        upper_triangular_mat : array-like, shape=[..., 3, 3]
            Upper triangular matrix.
        """
        n_points = gs.ndim(point)

        element_02 = point[..., 2] + 1 / 2 * point[..., 0] * point[..., 1]

        if n_points == 1:
            modified_point = gs.array([1, point[0], element_02, 1, point[1], 1])
        else:
            modified_point = gs.stack(
                (
                    gs.ones(n_points),
                    point[..., 0],
                    element_02,
                    gs.ones(n_points),
                    point[..., 1],
                    gs.ones(n_points),
                ),
                axis=1,
            )

        return gs.triu(SymmetricMatrices.from_vector(modified_point))
