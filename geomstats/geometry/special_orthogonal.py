"""Exposes the `SpecialOrthogonal` group class."""

from geomstats.geometry.general_linear import GeneralLinear


class SpecialOrthogonal(GeneralLinear):
    """Class for special orthogonal groups."""

    def __init__(self, n):
        self.n = n
        self.dim = int((n * (n - 1)) / 2)

    def belongs(self, point):
        """Check whether point is an orthogonal matrix."""
        return self.equal(
            self.mul(point, self.transpose(point)),
            self.identity())

    @classmethod
    def inv(cls, point):
        """Return the transpose matrix of point."""
        return cls.transpose(point)

    @classmethod
    def is_tangent(cls, vector):
        """Check whether vector is a skew-symmetric matrix."""
        return cls.equal(
            cls.transpose(vector),
            - vector)

    @classmethod
    def to_tangent(cls, vector):
        """Project vector onto skew-symmetric matrices."""
        return (cls.transpose(vector) - vector) / 2
