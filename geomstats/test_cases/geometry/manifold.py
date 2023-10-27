import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data


class _ManifoldTestCaseMixins:
    # TODO: check default_coords_type correcteness if intrinsic by comparing
    # with point shape?
    tangent_to_multiple = False

    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_dim(self, expected):
        self.assertEqual(self.space.dim, expected)

    def test_belongs(self, point, expected, atol):
        res = self.space.belongs(point, atol=atol)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_not_belongs(self, n_points, atol):
        shape = [val + 1 for val in self.space.shape]

        batch_shape = [n_points] if n_points else []
        point = gs.ones(batch_shape + shape)
        expected = gs.zeros(n_points, dtype=bool)

        self.test_belongs(point, expected, atol)

    @pytest.mark.random
    def test_random_point_belongs(self, n_points, atol):
        """Check random point belongs to manifold.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point = self.data_generator.random_point(n_points)
        expected = gs.ones(n_points, dtype=bool)

        self.test_belongs(point, expected, atol)

    @pytest.mark.shape
    def test_random_point_shape(self, n_points):
        """Check random point shape.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        """
        point = self.data_generator.random_point(n_points)

        expected_ndim = self.space.point_ndim + int(n_points > 1)
        self.assertEqual(gs.ndim(point), expected_ndim)

        self.assertAllEqual(gs.shape(point)[-self.space.point_ndim :], self.space.shape)

        if n_points > 1:
            self.assertEqual(gs.shape(point)[0], n_points)

    def test_is_tangent(self, vector, base_point, expected, atol):
        res = self.space.is_tangent(vector, base_point, atol=atol)
        self.assertAllEqual(res, expected)

    def test_to_tangent(self, vector, base_point, expected, atol):
        tangent_vec = self.space.to_tangent(vector, base_point)
        self.assertAllClose(tangent_vec, expected, atol=atol)

    @pytest.mark.vec
    def test_to_tangent_vec(self, n_reps, atol):
        # TODO: check if it makes sense
        vec = self.data_generator.random_point()
        point = self.data_generator.random_point()

        res = self.space.to_tangent(vec, point)

        vec_data = generate_vectorization_data(
            data=[dict(vector=vec, base_point=point, expected=res, atol=atol)],
            arg_names=["vector", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_to_tangent_is_tangent(self, n_points, atol):
        """Check to_tangent returns tangent vector.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(point)

        expected = gs.ones(n_points, dtype=bool)

        self.test_is_tangent(tangent_vec, point, expected, atol)

    def test_regularize(self, point, expected, atol):
        regularized_point = self.space.regularize(point)
        self.assertAllClose(regularized_point, expected, atol=atol)

    @pytest.mark.random
    def test_regularize_belongs(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        regularized_point = self.space.regularize(point)

        expected = gs.ones(n_points, dtype=bool)

        self.test_belongs(regularized_point, expected, atol)

    @pytest.mark.random
    def test_random_tangent_vec_is_tangent(self, n_points, atol):
        """Check to_tangent returns tangent vector.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = gs.ones(n_points, dtype=bool)
        self.test_is_tangent(tangent_vec, base_point, expected, atol)

    @pytest.mark.shape
    def test_random_tangent_vec_shape(self, n_points):
        """Check random point shape.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        """
        point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(point)

        expected_ndim = self.space.point_ndim + int(n_points > 1)
        self.assertEqual(gs.ndim(point), expected_ndim)

        self.assertAllEqual(
            gs.shape(tangent_vec)[-self.space.point_ndim :], self.space.shape
        )

        if n_points > 1:
            self.assertEqual(gs.shape(tangent_vec)[0], n_points)


class ManifoldTestCase(_ManifoldTestCaseMixins, TestCase):
    pass
