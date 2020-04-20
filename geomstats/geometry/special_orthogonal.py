"""Exposes the `SpecialOrthogonal` group class."""

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.lie_group import LieGroup
from geomstats.geometry.skew_symmetric_matrices import SkewSymmetricMatrices


class SpecialOrthogonal(GeneralLinear, LieGroup):
    """Class for special orthogonal groups."""

    def __init__(self, n):
        super(SpecialOrthogonal, self).__init__(
            dim=int((n * (n - 1)) / 2), default_point_type='matrix', n=n)
        self.lie_algebra = SkewSymmetricMatrices(n=n)

    def belongs(self, point):
        """Check whether point is an orthogonal matrix."""
        return self.equal(
            self.mul(point, self.transpose(point)), self.identity)

    @classmethod
    def inverse(cls, point):
        """Return the transpose matrix of point."""
        return cls.transpose(point)

    def _is_in_lie_algebra(self, tangent_vec):
        return self.lie_algebra.belongs(tangent_vec)

    @classmethod
    def _to_lie_algebra(cls, tangent_vec):
        """Project vector onto skew-symmetric matrices."""
        return cls.make_skew_symmetric(tangent_vec)

    def random_uniform(self, n_samples=1, tol=1e-6):
        if n_samples == 1:
            random_mat = gs.random.rand(self.n, self.n)
        else:
            random_mat = gs.random.rand(n_samples, self.n, self.n)
        skew = self.to_tangent(random_mat)
        return self.exp(skew)
