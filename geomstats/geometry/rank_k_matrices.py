"""Module exposing the rank k euclidean matrices"""

import numpy as np


import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.matrices import Matrices


class RankKMatrices(OpenSet):
    """Class for the matrices with euclidean entries and rank k

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: m x n
    m : int
        Integer representing the shape of the matrices: m x n
    k : int
        Integer representing the rank of the matrices

    """

    def __init__(self, m, n, k, **kwargs):
        if 'dim' not in kwargs.keys():
            kwargs['dim'] = m * n
        super(FullRankMatrices, self).__init__(
            ambient_space=Matrices(m, n), ambient, **kwargs)
        self.rank=k

    def belongs(self, point):
        """Check if the matrix belongs to R_*^m*n, i.e. is full rank

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to be checked.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if point is in R_*^m*n
        """
        has_right_size = self.ambient_space.belongs(point)
        if gs.all(has_right_size):
            #ANNA which is the right way to add numpy.matrix_rank to the backend functions
            rank = np.linalg.matrix_rank(point)
            return True if rank==self.rank else False
        return has_right_size

    # ANNA following general linear structure, I don't see any check in the dimension of the
    # input point, is it ok?
    # ANNA: is it the space of matrices of rank k dense in the space of matrices?
    def projection(self, point):
        r"""Project a matrix to the set of full rank matrices

        As the space of rank k matrices is dense in the space of matrices,
        this is not a projection per se, but a regularization if the matrix input X
        is not already full rank: `math:`X + \epsilon [I_rank, 0]` is returned
        where :math:`\epsilon=gs.atol`

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., dim_embedding]
            Projected point.
        """

        # ANNA I substitute self.identity with np.identity with gs.eye
        belongs = self.belongs(point)
        regularization = gs.einsum(
            '...,ij->...ij', gs.where(~belongs, gs.atol, 0.),
            gs.eye(self.ambient_space.shape[0], self.ambient_space.shape[1]))
        projected = point + regularization
        # ANNA now it gives a mistake if there dimension of the input mat is not the embedding space dimension
        return projected

    def random_point(self, n_samples=1, bound=1., n_iter=100):
        """Sample in R_*^m*n from the uniform distribution

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound: float
            Bound of the interval in which to sample each matrix entry.
            Optional, default: 1.
        n_iter : int
            Maximum number of trials to sample a matrix with full rank
            Optional, default: 100.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Point sampled on R_*^m*n
        """
        m = self.ambient_space.shape[0]
        n = self.ambient_space.shape[1]
        sample = []
        n_accepted, iteration = 0, 0
        criterion_func = lambda x: x==self.rank
        while n_accepted < n_samples and iteration < n_iter:
            raw_samples = gs.random.normal(size=(n_samples - n_accepted, m, n))
            # ANNA here using again numpy.linalg.matrix_rank, remember to substitute it when loagind from backend
            ranks = np.linalg.matrix_rank(raw_samples)
            selected = criterion_func(ranks)
            sample.append(raw_samples[selected])
            n_accepted += gs.sum(selected)
            iteration += 1
        if n_samples == 1:
            return sample[0][0]
        return gs.concatenate(sample)


# QUI
    @classmethod
    def orbit(cls, point, base_point=None):
        r"""
        Compute the one-parameter orbit of base_point passing through point.

        Parameters
        ----------
        point : array-like, shape=[n, n]
            Target point.
        base_point : array-like, shape=[n, n], optional
            Base point.
            Optional, defaults to identity if None.

        Returns
        -------
        path : callable
            One-parameter orbit.
            Satisfies `path(0) = base_point` and `path(1) = point`.

        Notes
        -----
        Denoting `point` by :math:`g` and `base_point` by :math:`h`,
        the orbit :math:`\gamma` satisfies:

        .. math::

            \gamma(t) = {\mathrm e}^{t X} \cdot h \\
            \quad \text{with} \quad\\
            {\mathrm e}^{X} = g h^{-1}

        The path is not uniquely defined and depends on the choice of :math:`V`
        returned by :py:meth:`log`.

        Vectorization
        -------------
        Return a collection of trajectories (4-D array)
        from a collection of input matrices (3-D array).
        """
        tangent_vec = cls.log(point, base_point)

        def path(time):
            vecs = gs.einsum('t,...ij->...tij', time, tangent_vec)
            return cls.exp(vecs, base_point)
        return path
