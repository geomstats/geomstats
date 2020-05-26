"""Module exposing the GeneralLinear group class."""

from itertools import product

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices


class GeneralLinear(Matrices):
    """Class for the general linear group GL(n).

    Parameters
    ----------
    n : int
        Integer representing the shape of the matrices: n x n.
    """

    def __init__(self, n, **kwargs):
        Matrices.__init__(self, n, n)

        self.n = n

    def belongs(self, point):
        """Check if a matrix is invertible and of the right shape.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to be checked.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean denoting if point is in GL(n).
        """
        point_shape = point.shape
        mat_dim_1, mat_dim_2 = point_shape[-2], point_shape[-1]
        det = gs.linalg.det(point)
        return gs.logical_and(
            mat_dim_1 == self.n and mat_dim_2 == self.n,
            det != 0.)

    def get_identity(self):
        """Return the identity matrix."""
        return gs.eye(self.n, self.n)
    identity = property(get_identity)

    @classmethod
    def compose(cls, *args):
        """Return the product of a collection of matrices."""
        return cls.mul(*args)

    @staticmethod
    def inverse(point):
        """Return the inverse of a matrix.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Matrix to be inverted.
        """
        return gs.linalg.inv(point)

    def _replace_values(self, samples, new_samples, indcs):
        """Replace samples with new samples at specific indices."""
        replaced_indices = [
            i for i, is_replaced in enumerate(indcs) if is_replaced]
        value_indices = list(
            product(replaced_indices, range(self.n), range(self.n)))
        return gs.assignment(samples, gs.flatten(new_samples), value_indices)

    def random_uniform(self, n_samples=1, tol=1e-6):
        """Sample in GL(n) from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        tol: float
            Threshold for the absolute value of the determinant of the
            returned matrix.
            Optional, default: 1e-6.

        Returns
        -------
        samples : array-like, shape=[..., n, n]
            Point sampled on GL(n).
        """
        samples = gs.random.rand(n_samples, self.n, self.n)
        while True:
            dets = gs.linalg.det(samples)
            indcs = gs.isclose(dets, 0.0, atol=tol)
            num_bad_samples = gs.sum(indcs)
            if num_bad_samples == 0:
                break
            new_samples = gs.random.rand(num_bad_samples, self.n, self.n)
            samples = self._replace_values(samples, new_samples, indcs)
        if n_samples == 1:
            samples = gs.squeeze(samples, axis=0)
        return samples

    @classmethod
    def exp(cls, tangent_vec, base_point=None):
        r"""
        Exponentiate a left-invariant vector field from a base point.

        The vector input is not an element of the Lie algebra, but of the
        tangent space at base_point: if :math:`g` denotes `base_point`,
        :math:`v` the tangent vector, and :math:'V = g^{-1} v' the associated
        Lie algebra vector, then

        .. math::

            \exp(v, g) = mul(g, \exp(V))

        Therefore, the Lie exponential is obtained when base_point is None, or
        the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., n, n]
            Tangent vector at base point.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, defaults to identity if None.

        Returns
        -------
        point : array-like, shape=[..., n, n]
            Left multiplication of `exp(algebra_mat)` with `base_point`.
        """
        expm = gs.linalg.expm
        if base_point is None:
            return expm(tangent_vec)
        lie_algebra_vec = cls.mul(cls.inverse(base_point), tangent_vec)
        return cls.mul(base_point, cls.exp(lie_algebra_vec))

    @classmethod
    def log(cls, point, base_point=None):
        r"""
        Compute a left-invariant vector field bringing base_point to point.

        The output is a vector of the tangent space at base_point, so not a Lie
        algebra element if it is not the identity.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.
            Optional, defaults to identity if None.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Matrix such that `exp(tangent_vec, base_point) = point`.

        Notes
        -----
        Denoting `point` by :math:`g` and `base_point` by :math:`h`,
        the output satisfies:

        .. math::

            g = \exp(\log(g, h), h)
        """
        logm = gs.linalg.logm
        if base_point is None:
            return logm(point)
        lie_algebra_vec = logm(cls.mul(cls.inverse(base_point), point))
        return cls.mul(base_point, lie_algebra_vec)

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
        # TODO (nina): Will work when expm gets properly 4-D vectorized.
        tangent_vec = cls.log(point, base_point)

        def path(time):
            vecs = gs.einsum('t,...ij->...tij', time, tangent_vec)
            return cls.exp(vecs, base_point)
        return path
