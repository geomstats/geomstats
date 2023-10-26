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

        expected_shape = (
            get_batch_shape(self.space, point) + (n_samples,) + self.space.support_shape
        )
        self.assertEqual(res.shape, expected_shape)

    def _check_sample_belongs_to_support(self, sample, atol):
        raise NotImplementedError("Need to define `_check_sample_belongs_to_support`.")

    @pytest.mark.random
    def test_sample_belongs_to_support(self, n_points, n_samples, atol):
        point = self.data_generator.random_point(n_points)
        sample = self.space.sample(point, n_samples)

        self._check_sample_belongs_to_support(sample, atol)

    def test_point_to_pdf(self, x, point, expected, atol):
        pdf = self.space.point_to_pdf(point)
        res = pdf(x)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_point_to_pdf_vec(self, n_reps, n_samples, atol):
        point = self.data_generator.random_point()

        x = self.space.sample(point, n_samples)
        expected = self.space.point_to_pdf(point)(x)

        vec_data = generate_vectorization_data(
            data=[dict(x=x, point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_point_to_pdf_against_scipy(self, n_points, n_samples, atol):
        point = self.data_generator.random_point(n_points)

        sample_point = point if n_points == 1 else point[0]
        sample = self.space.sample(sample_point, n_samples=n_samples)
        print(sample.shape)

        res = self.space.point_to_pdf(point)(sample)
        res_ = self.random_variable.pdf(sample, point)

        self.assertAllClose(res, res_, atol=atol)

    def test_point_to_cdf(self, x, point, expected, atol):
        cdf = self.space.point_to_cdf(point)
        res = cdf(x)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_point_to_cdf_vec(self, n_reps, n_samples, atol):
        point = self.data_generator.random_point()

        x = self.space.sample(point, n_samples)
        expected = self.space.point_to_cdf(point)(x)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_point_to_cdf_bounds(self, n_points, n_samples):
        point = self.data_generator.random_point(n_points)
        pdf = self.space.point_to_cdf(point)

        sample_point = point if n_points == 1 else point[0]
        sample = self.space.sample(sample_point, n_samples)
        res = pdf(sample)
        self.assertTrue(gs.all(res >= 0.0))
        self.assertTrue(gs.all(res <= 1.0))
