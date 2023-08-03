import pytest

import geomstats.backend as gs
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class MultinomialDistributionsTestCase(
    InformationManifoldMixinTestCase, LevelSetTestCase
):
    pass


class MultinomialMetricTestCase(RiemannianMetricTestCase):
    def test_simplex_to_sphere(self, point, expected, atol):
        res = self.space.metric.simplex_to_sphere(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_simplex_to_sphere_belongs(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        point_sphere = self.space.metric.simplex_to_sphere(point)

        res = self.space.metric._sphere.belongs(point_sphere, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_sphere_to_simplex(self, point, expected, atol):
        res = self.space.metric.sphere_to_simplex(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_sphere_to_simplex_vec(self, n_reps, atol):
        point = self.data_generator_sphere.random_point()

        expected = self.space.metric.sphere_to_simplex(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_sphere_to_simplex_belongs(self, n_points, atol):
        point = self.data_generator_sphere.random_point(n_points)

        point_simplex = self.space.metric.sphere_to_simplex(point)
        expected = gs.ones(n_points, dtype=bool)

        res = self.space.belongs(point_simplex, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_sphere_to_simplex_after_simplex_to_sphere(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        point_sphere = self.space.metric.simplex_to_sphere(point)
        point_ = self.space.metric.sphere_to_simplex(point_sphere)

        self.assertAllClose(point_, point, atol=atol)

    @pytest.mark.random
    def test_simplex_to_sphere_after_sphere_to_simplex(self, n_points, atol):
        point_sphere = self.data_generator_sphere.random_point(n_points)

        point = self.space.metric.sphere_to_simplex(point_sphere)
        point_sphere_ = self.space.metric.simplex_to_sphere(point)

        self.assertAllClose(point_sphere_, gs.abs(point_sphere), atol=atol)

    def test_tangent_simplex_to_sphere(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.tangent_simplex_to_sphere(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_tangent_simplex_to_sphere_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        tangent_vec_sphere = self.space.metric.tangent_simplex_to_sphere(
            tangent_vec, base_point
        )
        base_point_sphere = self.space.metric.simplex_to_sphere(base_point)

        is_tangent = self.space.metric._sphere.is_tangent(
            tangent_vec_sphere, base_point_sphere, atol=atol
        )
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_tangent, expected)

    def test_tangent_sphere_to_simplex(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.tangent_sphere_to_simplex(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_sphere_to_simplex_vec(self, n_reps, atol):
        base_point = self.data_generator_sphere.random_point()
        tangent_vec = self.data_generator_sphere.random_tangent_vec(base_point)

        expected = self.space.metric.tangent_sphere_to_simplex(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tangent_sphere_to_simplex_is_tangent(self, n_points, atol):
        base_point_sphere = self.data_generator_sphere.random_point(n_points)
        tangent_vec_sphere = self.data_generator_sphere.random_tangent_vec(
            base_point_sphere
        )

        tangent_vec = self.space.metric.tangent_sphere_to_simplex(
            tangent_vec_sphere, base_point_sphere
        )
        base_point = self.space.metric.sphere_to_simplex(base_point_sphere)

        is_tangent = self.space.is_tangent(tangent_vec, base_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_tangent, expected)

    @pytest.mark.random
    def test_tangent_sphere_to_simplex_after_tangent_simplex_to_sphere(
        self, n_points, atol
    ):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        tangent_vec_sphere = self.space.metric.tangent_simplex_to_sphere(
            tangent_vec, base_point
        )
        base_point_sphere = self.space.metric.simplex_to_sphere(base_point)

        tangent_vec_ = self.space.metric.tangent_sphere_to_simplex(
            tangent_vec_sphere, base_point_sphere
        )
        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    @pytest.mark.random
    def test_tangent_simplex_to_sphere_after_tangent_sphere_to_simplex(
        self, n_points, atol
    ):
        base_point_sphere = gs.abs(self.data_generator_sphere.random_point(n_points))
        tangent_vec_sphere = self.data_generator_sphere.random_tangent_vec(
            base_point_sphere
        )

        tangent_vec = self.space.metric.tangent_sphere_to_simplex(
            tangent_vec_sphere, base_point_sphere
        )
        base_point = self.space.metric.sphere_to_simplex(base_point_sphere)

        tangent_vec_sphere_ = self.space.metric.tangent_simplex_to_sphere(
            tangent_vec, base_point
        )

        self.assertAllClose(tangent_vec_sphere_, tangent_vec_sphere, atol=atol)
