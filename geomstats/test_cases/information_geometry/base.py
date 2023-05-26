import random

import pytest

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.vectorization import get_batch_shape


class InformationManifoldMixinTestCase(TestCase):
    @pytest.mark.shape
    def test_sample_shape(self, n_points, n_samples):
        point = self.data_generator.random_point(n_points)

        res = self.space.sample(point, n_samples=n_samples)

        shape = res.shape
        expected_shape = get_batch_shape(self.space, point) + (
            n_samples,
            self.space.dim,
        )
        self.assertEqual(shape, expected_shape)

    def test_point_to_pdf(self, x, point, expected, atol):
        pdf = self.space.point_to_pdf(point)
        res = pdf(x)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_point_to_pdf_vec(self, n_reps, atol):
        # TODO: should this be a shape test instead?
        point = self.data_generator.random_point()

        # TODO: fix here
        x = gs.random.uniform(size=(random.randint(1, 4), self.space.dim))
        expected = self.space.point_to_pdf(point)(x)

        vec_data = generate_vectorization_data(
            data=[dict(x=x, point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_point_to_cdf(self, x, point, expected, atol):
        cdf = self.space.point_to_cdf(point)
        res = cdf(x)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_point_to_cdf_vec(self, n_reps, atol):
        # TODO: should this be a shape test instead?
        point = self.data_generator.random_point()

        # TODO: fix here
        x = gs.random.rand((random.randint(1, 4), self.space.dim))
        expected = self.space.point_to_cdf(point)(x)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
