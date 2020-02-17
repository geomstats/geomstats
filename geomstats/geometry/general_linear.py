"""Module exposing the GeneralLinear group class."""

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices


class GeneralLinear(Matrices):
    """Class for the general linear group GL(n)."""

    def __init__(self, n):
        Matrices.__init__(self, n, n)

    @staticmethod
    def belongs(point):
        """Test if a matrix is invertible."""
        det = gs.linalg.det(point)
        return gs.where(det != 0., True, False)

    def identity(self):
        """Return the identity matrix."""
        return gs.eye(self.n, self.n)

    @classmethod
    def compose(cls, *args):
        """Return the product of a collection of matrices."""
        return cls.mul(*args)

    @staticmethod
    def inv(point):
        """Return the inverse of a matrix."""
        return gs.linalg.inv(point)

    @classmethod
    def exp(cls, tangent_vec, base_point=None):
        """
        Exponentiate a left-invariant vector field from a base point.

        Parameters
        ----------
        tangent_vec :   array-like, shape=[..., n, n]
        base_point :    array-like, shape=[..., n, n]
            Defaults to identity.

        Returns
        -------
        point :         array-like, shape=[..., n, n]
            The left multiplication of `exp(tangent_vec)` with
            `base_point`.
        """
        expm = gs.linalg.expm
        if base_point is None:
            return expm(tangent_vec)
        else:
            return cls.mul(expm(tangent_vec), base_point)

    @classmethod
    def log(cls, point, base_point=None):
        """
        Calculate a left-invariant vector field bringing base_point to point.

        Parameters
        ----------
        point :         array-like, shape=[..., n, n]
        base_point :    array-like, shape=[..., n, n]
            Defaults to identity.

        Returns
        -------
        tangent_vec :   array-like, shape=[..., n, n]
            A matrix such that `exp(tangent_vec, base_point) = point`.
        """
        logm = gs.linalg.logm
        if base_point is None:
            return logm(point)
        else:
            return logm(cls.mul(point, cls.inv(base_point)))

    @classmethod
    def orbit(cls, point, base_point=None):
        r"""
        Compute the one-parameter orbit of base_point passing through point.

        Parameters
        ----------
        point : array-like, shape=[n, n]
            Target point.
        base_point : array-like, shape=[n, n], optional
            Base point. Defaults to identity.

        Returns
        -------
        path : callable
            The one-parameter orbit.
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

        Will work when expm gets properly 4-D vectorized.
        """
        tangent_vec = cls.log(point, base_point)

        def path(time):
            vecs = gs.einsum('t,...ij->...tij', time, tangent_vec)
            return cls.exp(vecs, base_point)
        return path
