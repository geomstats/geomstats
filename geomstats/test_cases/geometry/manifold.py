import geomstats.backend as gs
from geomstats.test.assertions import assert_belongs, assert_is_tangent
from geomstats.test.test_case import TestCase


class _ManifoldTestCaseMixins:
    # TODO: check default_coords_type correcteness if intrinsic by comparing
    # with point shape?
    # TODO: need to review to_tangent

    def test_dim(self, expected):
        self.assertEqual(self.space.dim, expected)

    def test_belongs(self, point, expected, atol=gs.atol):
        res = self.space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_belongs_is_true(self, point, atol=gs.atol):
        assert_belongs(self.space, point, atol=atol)

    def test_is_tangent(self, vector, base_point, expected, atol=gs.atol):
        res = self.space.is_tangent(vector, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_assert_is_tangent(self, vector, base_point, atol=gs.atol):
        """Check to_tangent returns tangent vector.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        assert_is_tangent(self.space, vector, base_point, atol=atol)

    def test_regularize(self, point, expected, atol=gs.atol):
        regularized_point = self.space.regularize(point)
        self.assertAllClose(regularized_point, expected, atol=atol)

    def test_regularize_belongs(self, point, atol=gs.atol):
        regularized_point = self.space.regularize(point)

        assert_belongs(self.space, regularized_point, atol=atol)


class ManifoldTestCase(_ManifoldTestCaseMixins, TestCase):
    pass
