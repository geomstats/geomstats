"""Exposes the `SpecialOrthogonal` group class."""

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear


class SpecialOrthogonal(GeneralLinear):
    """Class for special orthogonal groups."""

    def __init__(self, n):
        self.n = n
        self.dim = int((n * (n - 1)) / 2)

    def belongs(self, point):
        """Check whether point is an orthogonal matrix."""
        return self.equal(
            self.mul(point, self.transpose(point)), self.identity)

    @classmethod
    def inverse(cls, point):
        """Return the transpose matrix of point."""
        return cls.transpose(point)

    @classmethod
    def is_tangent(cls, vector):
        """Check whether vector is a skew-symmetric matrix."""
        return cls.equal(cls.transpose(vector), - vector)

    @classmethod
    def to_tangent(cls, vector):
        """Project vector onto skew-symmetric matrices."""
        return (cls.transpose(vector) - vector) / 2

    def random_uniform(self, n_samples=1, tol=1e-6):
        random_mat = gs.random.rand(n_samples, self.n, self.n)
        skew = self.to_tangent(random_mat)
        return self.exp(skew)
