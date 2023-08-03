import pytest

import geomstats.backend as gs
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class PullbackDiffeoMetricTestCase(RiemannianMetricTestCase):
    def test_diffeomorphism(self, base_point, expected, atol):
        res = self.space.metric.diffeomorphism(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_diffeomorphism_belongs(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        image_point = self.space.metric.diffeomorphism(base_point)

        belongs = self.space.metric.embedding_space.belongs(image_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(belongs, expected)

    def test_inverse_diffeomorphism(self, image_point, expected, atol):
        res = self.space.metric.inverse_diffeomorphism(image_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_diffeomorphism_vec(self, n_reps, atol):
        image_point = self.data_generator_embedding.random_point()

        expected = self.space.metric.inverse_diffeomorphism(image_point)

        vec_data = generate_vectorization_data(
            data=[dict(image_point=image_point, expected=expected, atol=atol)],
            arg_names=["image_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_inverse_diffeomorphism_belongs(self, n_points, atol):
        image_point = self.data_generator_embedding.random_point(n_points)

        point = self.space.metric.inverse_diffeomorphism(image_point)

        belongs = self.space.belongs(point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(belongs, expected)

    @pytest.mark.random
    def test_inverse_diffeomorphism_after_diffeomorphism(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        image_point = self.space.metric.diffeomorphism(base_point)
        base_point_ = self.space.metric.inverse_diffeomorphism(image_point)

        self.assertAllClose(base_point_, base_point, atol=atol)

    @pytest.mark.random
    def test_diffeomorphism_after_inverse_diffeomorphism(self, n_points, atol):
        image_point = self.data_generator_embedding.random_point(n_points)

        point = self.space.metric.inverse_diffeomorphism(image_point)
        image_point_ = self.space.metric.diffeomorphism(point)

        self.assertAllClose(image_point_, image_point, atol=atol)

    def test_jacobian_diffeomorphism(self, base_point, expected, atol):
        res = self.space.metric.jacobian_diffeomorphism(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_tangent_diffeomorphism(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.tangent_diffeomorphism(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_tangent_diffeomorphism_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        image_tangent_vec = self.space.metric.tangent_diffeomorphism(
            tangent_vec, base_point
        )
        image_point = self.space.metric.diffeomorphism(base_point)

        is_tangent = self.space.metric.embedding_space.is_tangent(
            image_tangent_vec, image_point, atol=atol
        )
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_tangent, expected)

    def test_inverse_jacobian_diffeomorphism(self, image_point, expected, atol):
        res = self.space.metric.inverse_jacobian_diffeomorphism(image_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_jacobian_diffeomorphism_vec(self, n_reps, atol):
        image_point = self.data_generator_embedding.random_point()

        expected = self.space.metric.inverse_jacobian_diffeomorphism(image_point)

        vec_data = generate_vectorization_data(
            data=[dict(image_point=image_point, expected=expected, atol=atol)],
            arg_names=["image_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_inverse_tangent_diffeomorphism(
        self, image_tangent_vec, image_point, expected, atol
    ):
        res = self.space.metric.inverse_tangent_diffeomorphism(
            image_tangent_vec,
            image_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_inverse_tangent_diffeomorphism_vec(self, n_reps, atol):
        image_point = self.data_generator_embedding.random_point()
        image_tangent_vec = self.data_generator_embedding.random_tangent_vec(
            image_point
        )

        expected = self.space.metric.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    image_tangent_vec=image_tangent_vec,
                    image_point=image_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["image_tangent_vec", "image_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_inverse_tangent_diffeomorphism_is_tangent(self, n_points, atol):
        image_point = self.data_generator_embedding.random_point(n_points)
        image_tangent_vec = self.data_generator_embedding.random_tangent_vec(
            image_point
        )

        tangent_vec = self.space.metric.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )
        base_point = self.space.metric.inverse_diffeomorphism(image_point)

        is_tangent = self.space.is_tangent(tangent_vec, base_point, atol=atol)
        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_tangent, expected)

    @pytest.mark.random
    def test_inverse_tangent_diffeomorphism_after_tangent_diffeomorphism(
        self, n_points, atol
    ):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        image_tangent_vec = self.space.metric.tangent_diffeomorphism(
            tangent_vec, base_point
        )
        image_point = self.space.metric.diffeomorphism(base_point)

        tangent_vec_ = self.space.metric.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )
        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    @pytest.mark.random
    def test_tangent_diffeomorphism_after_inverse_tangent_diffeomorphism(
        self, n_points, atol
    ):
        image_point = self.data_generator_embedding.random_point(n_points)
        image_tangent_vec = self.data_generator_embedding.random_tangent_vec(
            image_point
        )

        tangent_vec = self.space.metric.inverse_tangent_diffeomorphism(
            image_tangent_vec, image_point
        )
        base_point = self.space.metric.diffeomorphism(image_point)

        image_tangent_vec_ = self.space.metric.tangent_diffeomorphism(
            tangent_vec, base_point
        )
        self.assertAllClose(image_tangent_vec_, image_tangent_vec, atol=atol)
