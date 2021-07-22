"""Module exposing the GeneralLinear group class."""

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.lie_group import MatrixLieGroup
from geomstats.geometry.matrices import Matrices


class GeneralLinear(MatrixLieGroup, OpenSet):
    """Class for the general linear group GL(n) and its identity component.

    If `positive_det=True`, this is the connected component of the identity,
    i.e. the space of matrices with positive determinant.

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    positive_det : bool
        Whether to restrict to the identity connected component of the
        general linear group, i.e. matrices with positive determinant.
        Optional, default: False.
    """

    def __init__(self, n, positive_det=False, **kwargs):
        if 'dim' not in kwargs.keys():
            kwargs['dim'] = n ** 2
        super(GeneralLinear, self).__init__(
            ambient_space=Matrices(n, n), n=n, **kwargs)
        self.positive_det = positive_det

    def projection(self, point):
        r"""Project a matrix to the general linear group.

        As GL(n) is dense in the space of matrices, this is not a projection
        per se, but a regularization if the matrix is not already invertible:
        :math:`X + \epsilon I_n` is returned where :math:`\epsilon=gs.atol`
        is returned for an input X.

        Parameters
        ----------
        point : array-like, shape=[..., dim_embedding]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., dim_embedding]
            Projected point.
        """
        belongs = self.belongs(point)
        regularization = gs.einsum(
            '...,ij->...ij', gs.where(~belongs, gs.atol, 0.), self.identity)
        projected = point + regularization
        if self.positive_det:
            det = gs.linalg.det(point)
            return utils.flip_determinant(projected, det)
        return projected

    def belongs(self, point, atol=gs.atol):
        """Check if a matrix is invertible and of the right shape.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol : float
            Tolerance threshold for the determinant.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if point is in GL(n).
        """
        has_right_size = self.ambient_space.belongs(point)
        if gs.all(has_right_size):
            det = gs.linalg.det(point)
            return det > atol if self.positive_det else gs.abs(det) > atol
        return has_right_size

    def random_point(self, n_samples=1, bound=1., n_iter=100):
        """Sample in GL(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound: float
            Bound of the interval in which to sample each matrix entry.
            Optional, default: 1.
        n_iter : int
            Maximum number of trials to sample a matrix with positive det.
            Optional, default: 100.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Point sampled on GL(n).
        """
        n = self.n
        sample = []
        n_accepted, iteration = 0, 0
        criterion_func = (lambda x: x) if self.positive_det else gs.abs
        while n_accepted < n_samples and iteration < n_iter:
            raw_samples = gs.random.normal(size=(n_samples - n_accepted, n, n))
            dets = gs.linalg.det(raw_samples)
            criterion = criterion_func(dets) > gs.atol
            if gs.any(criterion):
                sample.append(raw_samples[criterion])
                n_accepted += gs.sum(criterion)
            iteration += 1
        if n_samples == 1:
            return sample[0]
        return gs.concatenate(sample)

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
