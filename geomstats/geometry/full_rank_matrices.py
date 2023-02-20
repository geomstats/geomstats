r"""Full rank Euclidean matrices :math:`R_*^{m\times n}`.

Lead author: Anna Calissano.
"""

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.matrices import Matrices, MatricesMetric


class FullRankMatrices(OpenSet):
    r"""Class for :math:`R_*^{n\times k}` matrices of dimension n x k and full rank.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x k
    k : int
        Integer representing the shape of the matrices: n x k
    """

    def __init__(self, n, k, **kwargs):
        kwargs.setdefault("dim", n * k)
        kwargs.setdefault("metric", MatricesMetric(n, k))
        super().__init__(embedding_space=Matrices(n, k), **kwargs)
        self.rank = min(n, k)
        self.n = n
        self.k = k

    def belongs(self, point, atol=gs.atol):
        r"""Check if the matrix belongs to :math:`R_*^{n \times k}`.

        Parameters
        ----------
        point : array-like, shape=[..., n, k]
            Matrix to be checked.
        atol : float
            Unused.

        Returns
        -------
        belongs : Boolean
            Denoting if point is in :math:`R_*^{m\times n}`.
        """
        has_right_size = self.embedding_space.belongs(point)
        has_right_rank = gs.where(
            gs.linalg.matrix_rank(point) == self.rank, True, False
        )
        return gs.logical_and(gs.array(has_right_size), has_right_rank)

    def projection(self, point):
        r"""Project a matrix to the set of full rank matrices.

        As the space of full rank matrices is dense in the space of matrices,
        this is not a projection per se, but a regularization if the matrix input X
        is not already full rank: :math:`X + \epsilon [I_{rank}, 0]` is returned
        where :math:`\epsilon=gs.atol`

        Parameters
        ----------
        point : array-like, shape=[..., n, k]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., n, k]
            Projected point.
        """
        belongs = self.belongs(point)
        regularization = gs.einsum(
            "...,ij->...ij",
            gs.where(~belongs, gs.atol, 0.0),
            gs.eye(self.embedding_space.shape[0], self.embedding_space.shape[1]),
        )
        projected = point + regularization
        return projected

    def random_point(self, n_samples=1, bound=1.0, n_iter=100):
        r"""Sample in :math:`R_*^{n\times k}` from a normal distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound: float
            This parameter is ignored.
        n_iter : int
            Maximum number of trials to sample a matrix with full rank.
            Optional, default: 100.

        Returns
        -------
        samples : array-like, shape=[..., n, k]
            Point sampled on :math:`R_*^{n\times k}`.
        """
        sample = []
        n_accepted, iteration = 0, 0
        while n_accepted < n_samples and iteration < n_iter:
            raw_samples = gs.random.normal(
                size=(n_samples - n_accepted, self.n, self.k)
            )
            ranks = gs.linalg.matrix_rank(raw_samples)
            selected = ranks == self.rank
            sample.append(raw_samples[selected])
            n_accepted += gs.sum(selected)
            iteration += 1
        if n_samples == 1:
            return sample[0][0]
        return gs.concatenate(sample)
