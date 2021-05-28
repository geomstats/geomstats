"""Module exposing the GeneralLinear group class."""

from itertools import product

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.lie_group import MatrixLieGroup
from geomstats.geometry.matrices import Matrices


class GeneralLinear(MatrixLieGroup, OpenSet):
    """Class for the general linear group GL(n).

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n, **kwargs):
        if 'dim' not in kwargs.keys():
            kwargs['dim'] = n ** 2
        super(GeneralLinear, self).__init__(
            ambient_space=Matrices(n, n), n=n, **kwargs)

    def projection(self, point):
        r"""Project a matrix to the general linear group.

        As GL(n) is dense in the space of matrices, this is not a projection
        per se, but a regularization if the matrix is not already invertible:
        :math: `X + \epsilon I_n` is returned where :math: `\epsilon=gs.atol`
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
        return projected

    def belongs(self, point, atol=gs.atol):
        """Check if a matrix is invertible and of the right shape.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to be checked.
        atol :  float
            Tolerance threshold for the determinant.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if point is in GL(n).
        """
        has_right_size = self.ambient_space.belongs(point)
        if gs.all(has_right_size):
            det = gs.linalg.det(point)
            return gs.abs(det) > atol
        return has_right_size

    def _replace_values(self, samples, new_samples, indcs):
        """Replace samples with new samples at specific indices."""
        replaced_indices = [
            i for i, is_replaced in enumerate(indcs) if is_replaced]
        value_indices = list(
            product(replaced_indices, range(self.n), range(self.n)))
        return gs.assignment(samples, gs.flatten(new_samples), value_indices)

    def random_point(self, n_samples=1, bound=1.):
        """Sample in GL(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound: float
            Bound of the interval in which to sample each matrix entry.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Point sampled on GL(n).
        """
        samples = gs.random.normal(size=(n_samples, self.n, self.n))
        while True:
            dets = gs.linalg.det(samples)
            indcs = gs.isclose(dets, 0.0)
            num_bad_samples = gs.sum(indcs)
            if num_bad_samples == 0:
                break
            new_samples = gs.random.normal(
                size=(num_bad_samples, self.n, self.n))
            samples = self._replace_values(samples, new_samples, indcs)
        if n_samples == 1:
            samples = gs.squeeze(samples, axis=0)
        return samples

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
            \quad {\mathrm with} \quad\\
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
