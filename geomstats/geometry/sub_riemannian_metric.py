import abc

import geomstats.backend as gs
import geomstats.geometry as geometry


class SubRiemannianMetric(abc.ABC):
    """Class for Sub-Riemannian metrics.

    This implementation assumes a distribution of constant dimension.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    dist_dim : int
        Dimension of the distribution
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    """

    def __init__(self, dim, dist_dim, default_point_type="vector"):
        super(SubRiemannianMetric, self).__init__(
            dim=dim, dist_dim=dist_dim, default_point_type=default_point_type
        )

    def metric_matrix(self, base_point):
        """Metric matrix at the tangent space at a base point.

        This is a sub-Riemannian metric, so it is assumed to satisfy the conditions
        of an inner product only on each distribution subspace.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        _ : array-like, shape=[..., dim, dim]
            Inner-product matrix.
        """
        raise NotImplementedError(
            "The computation of the metric matrix" " is not implemented."
        )

    @abc.abstractmethod
    def frame(self, point):
        """Frame field for the distribution.

        The frame field spans the distribution at 'point'.The frame field is
        represented as a matrix, whose columns are the frame field vectors.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            point.
            Optional, default: None.

        Returns
        -------
        _ : array-like, shape=[..., dim, dist_dim]
            Frame field matrix.
        """

    def cometric_sub_matrix(self, basepoint):
        """Cometric  sub matrix of dimension dist_dim x dist_dim.

        Let {X_i}, i = 1, .., dist_dim, be an arbitrary frame for the distribution
        and let g be the sub-Riemannian metric. Then cometric_sub_matrix is the
        matrix given by the inverse of the matrix g_ij = g(X_i, X_j),
        where i,j = 1, .., dist_dim.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        _ : array-like, shape=[..., dist_dim, dist_dim]
            Cometric submatrix.
        """
        raise NotImplementedError(
            "The computation of the cometric submatrix" " is not implemented."
        )

    def hamiltonian(self, state):
        """Sub-Riemannian Hamiltonian.

        Parameters
        ----------
        state : array-like, shape=[..., 2*dim]
            Base point.
            Optional, default: None.

        Returns
        -------
        _ : array-like, shape=[..., 1]
            Hamiltonian evaluated at state.
        """
        raise NotImplementedError(
            "The computation of the hamiltonian" " is not implemented."
        )
