"""Implementation of the 3D Heisenberg group."""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.lie_group import LieGroup


class HeisenbergVectors(LieGroup):
    r"""Class for the 3D Heisenberg group in vector representation.

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
            dim=int(3), default_point_type='vector',
            lie_algebra=Euclidean(3))

    def get_identity(self, point_type='vector'):
        """Get the identity of the 3D Heisenberg group.

        Parameters
        ----------
        point_type : str, {'vector', 'matrix'}
            Point_type of the returned value. Unused here.

        Returns
        -------
        get_identity : array-like, shape=[3,]
        """
        return gs.zeros(self.dim)

    identity = property(get_identity)

    def belongs(self, point):
        """Evaluate if a point belongs to the 3D Heisenberg group.

        Parameters
        ----------
        point : array-like, shape=[..., 3]
            Point to evaluate.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the Heisenberg
            group (i.e. R^3 in this representation).
        """
        point_dim = point.shape[-1]
        belongs = point_dim == self.dim
        if gs.ndim(point) == 2:
            belongs = gs.tile([belongs], (point.shape[0],))

        return belongs

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
        composed : array-like, shape=[..., 3]
            Product of point_a and point_b along the first dimension.
        """
        point_ab = point_a + point_b

        point_ab_additional_term = 1 / 2 * (point_a[..., 0] *
                                            point_b[1] - point_a[..., 1] *
                                            point_b[0])

        point_ab[..., 2] = point_ab[..., 2] + point_ab_additional_term
        return point_ab

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

    def jacobian_translation(
            self, point, left_or_right='left'):
        """Compute the Jacobian of left/right translation by a point.

        Compute the Jacobian matrix of the left translation by the point.

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
        jacobian : array-like, shape=[..., 3, 3]
            Jacobian of the left/right translation by point.
        """
        if left_or_right == 'left':
            return(gs.array([[1, 0, 0],
                             [0, 1, 0],
                             [-point[1] / 2, point[0] / 2, 1]]))
        return(gs.array([[1, 0, 0],
                         [0, 1, 0],
                         [point[1] / 2, -point[0] / 2, 1]]))

    def random_point(self, n_samples=1, bound=1.):
        """Sample in the Euclidean space R^3 with a uniform distribution in a box.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of hypercube support of the uniform distribution.
            Optional, default: 1.0

        Returns
        -------
        point : array-like, shape=[..., 3]
           Sample.
        """
        size = (self.dim,)
        if n_samples != 1:
            size = (n_samples, self.dim)
        point = bound * (gs.random.rand(*size) - 0.5) * 2

        return point

    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of the tangent vector at the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 3]
            Tangent vector at base point.

        Returns
        -------
        point : array-like, shape=[..., 3]
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
        tangent_vec : array-like, shape=[..., 3]
            Group logarithm.
        """
        return point

    @staticmethod
    def upperTriangular_matrix_from_vector(point):
        """Compute the upper triangular matrix corresponding to the vector.

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
        return gs.array([[1, point[0], point[2] + 1 / 2 * point[0] * point[1]],
                         [0, 1, point[1]],
                         [0, 0, 1]])
