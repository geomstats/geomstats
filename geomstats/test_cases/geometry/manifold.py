import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.vectorization import get_batch_shape


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
        batch_shape = get_batch_shape(self.space.point_ndim, point)
        expected = gs.ones(batch_shape, dtype=bool)

        self.test_belongs(point, expected, atol)

    def test_is_tangent(self, vector, base_point, expected, atol=gs.atol):
        res = self.space.is_tangent(vector, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_assert_is_tangent(self, tangent_vec, base_point, atol):
        """Check to_tangent returns tangent vector.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        batch_shape = get_batch_shape(self.space.point_ndim, base_point, tangent_vec)
        expected = gs.ones(batch_shape, dtype=bool)
        self.test_is_tangent(tangent_vec, base_point, expected, atol)

    def test_regularize(self, point, expected, atol=gs.atol):
        regularized_point = self.space.regularize(point)
        self.assertAllClose(regularized_point, expected, atol=atol)

    def test_regularize_belongs(self, point, atol=gs.atol):
        regularized_point = self.space.regularize(point)

        batch_shape = get_batch_shape(self.space.point_ndim, point)
        expected = gs.ones(batch_shape, dtype=bool)

        self.test_belongs(regularized_point, expected, atol)


class ManifoldTestCase(_ManifoldTestCaseMixins, TestCase):
    pass
