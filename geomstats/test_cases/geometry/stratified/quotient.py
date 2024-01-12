import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator, get_random_times
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.stratified.point_set import (
    PointSetMetricWithArrayTestCase,
)


class AlignerAlgorithmTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.total_space)

    def test_align(self, point, base_point, expected, atol):
        res = self.aligner.align(self.total_space, point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_align_vec(self, n_reps, atol):
        # TODO: extend vec_test generation to accept this case?
        point = self.data_generator.random_point()
        base_point = self.data_generator.random_point()

        expected = self.aligner.align(self.total_space, point, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)


class AlignerAlgorithmCmpTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.total_space)

    @pytest.mark.random
    def test_align(self, n_points, atol):
        point = self.data_generator.random_point(n_points)
        base_point = self.data_generator.random_point(n_points)

        total_space = self.total_space
        aligned = self.aligner.align(total_space, point, base_point)
        other_aligned = self.other_aligner.align(total_space, point, base_point)

        self.assertAllClose(aligned, other_aligned, atol=atol)


class QuotientMetricWithArrayTestCase(PointSetMetricWithArrayTestCase):
    """Quotient metric test cases.

    Applies to discrete group actions.
    """

    @pytest.mark.random
    def test_geodesic_boundary_points(self, n_points, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        time = gs.array([0.0, 1.0])

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)

        res = geod_func(time)
        aligned_end_point = self.total_space.aligner.align(end_point, initial_point)
        expected = gs.stack(
            [initial_point, aligned_end_point], axis=-(self.space.point_ndim + 1)
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_geodesic_bvp_reverse(self, n_points, n_times, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        time = get_random_times(n_times)

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        geod_func_reverse = self.space.metric.geodesic(
            end_point, end_point=initial_point
        )

        res = geod_func(time)
        res_ = geod_func_reverse(1.0 - time)
        if n_points > 1:
            aligned_res_ = gs.stack(
                [
                    self.total_space.aligner.align(point_res_, initial_point_)
                    for point_res_, initial_point_ in zip(res_, initial_point)
                ]
            )
        else:
            aligned_res_ = self.total_space.aligner.align(res_, initial_point)

        self.assertAllClose(res, aligned_res_, atol=atol)
