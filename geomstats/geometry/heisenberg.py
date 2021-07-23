"""Implementation of the 3D Heisenberg group"""

import geomstats.backend as gs
from geomstats.geometry.lie_group import LieGroup


class heisenbergVectors(LieGroup):
    r"""Class for the 3D Heisenberg group in vector representation.

    The 3D Heisenberg group represented as R^3. It is a Carnot group. It can be
    equipped with a natural sub-Riemannian structure, making it a fundamental
    example in sub-Riemannian geometry.

    Parameters: none
    """

    def __init__(self, **kwargs):
        super(heisenbergVectors, self).__init__(dim=int(3),
                                                default_point_type='vector')

    def belongs(self, point):
        """Evaluate if a point belongs to MyManifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point to evaluate.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the Heisenberg
            group (in this case R^3).
        """
        # Perform operations to check if point belongs
        # to the manifold, for example:
        belongs = point.shape[-1] == self.dim
        return belongs

    def get_identity(self, point_type='vector'):
        """Get the identity of the group.

        Parameters
        ----------
        point_type : str, {'vector', 'matrix'}
            Point_type of the returned value. Unused here.

        Returns
        -------
        get_identity : array-like, shape=[1,]
        """
        return gs.zeros(self.dim)

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
        return (point_a + point_b +
                1 / 2 * gs.array([0, 0, point_a[0] *
                                  point_b[1] - point_a[1] *
                                  point_b[0]]))

    def inverse(self, point):
        """Compute the group inverse of point.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point.

        Returns
        -------
        inv_point : array-like, shape=[..., 3]
            Inverse.
        """
        return -point

    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of the tangent vector at the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dimension]
            Tangent vector at base point.

        Returns
        -------
        point : array-like, shape=[..., dimension]
            Point.
        """
        return tangent_vec

    def log_from_identity(self, point):
        """Compute the group logarithm of the point at the identity.

        Parameters
        ----------
        point : array-like, shape=[..., dimension]
            Point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dimension]
            Group logarithm.
        """
        return point

    def upperTriangular_matrix_from_vector(self, vec):
        """Get the upper triangular matrix corresponding to the vector.

        Parameters
        ----------
        vec : array-like, shape=[..., dim]
            Vector.

        Returns
        -------
        skew_mat : array-like, shape=[..., n, n]
            Skew-symmetric matrix.
        """
        return gs.array([[1, vec[0], vec[2] + 1 / 2 * vec[0] * vec[1]],
                         [0, 1, vec[1]],
                         [0, 0, 1]])
