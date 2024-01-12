import pytest

import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data


class PointToGeodesicAlignerTestCase(TestCase):
    def test_align(self, geodesic, point, expected, atol):
        res = self.aligner.align(self.total_space, geodesic, point)
        self.assertAllClose(res, expected, atol=atol)

    def test_align_with_endpoints(
        self, initial_point, end_point, point, expected, atol
    ):
        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        self.test_align(geodesic, point, expected, atol)

    @pytest.mark.vec
    def test_align_vec(self, n_reps, atol):
        initial_point = self.data_generator.random_point()
        end_point = self.data_generator.random_point()

        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        point = self.data_generator.random_point()

        expected = self.aligner.align(self.total_space, geodesic, point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    initial_point=initial_point,
                    end_point=end_point,
                    point=point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["initial_point", "end_point", "point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data, test_fnc_name="test_align_with_endpoints")

    def test_dist(self, geodesic, point, expected, atol):
        res = self.aligner.dist(self.total_space, geodesic, point)
        self.assertAllClose(res, expected, atol=atol)

    def test_dist_with_endpoints(self, initial_point, end_point, point, expected, atol):
        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        self.test_dist(geodesic, point, expected, atol)

    @pytest.mark.vec
    def test_dist_vec(self, n_reps, atol):
        initial_point = self.data_generator.random_point()
        end_point = self.data_generator.random_point()

        geodesic = self.total_space.quotient.metric.geodesic(initial_point, end_point)
        point = self.data_generator.random_point()

        expected = self.aligner.dist(self.total_space, geodesic, point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    initial_point=initial_point,
                    end_point=end_point,
                    point=point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["initial_point", "end_point", "point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data, test_fnc_name="test_dist_with_endpoints")

    @pytest.mark.random
    def test_dist_along_geodesic_is_zero(self, n_points, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        geodesic = self.total_space.quotient.metric.geodesic(
            initial_point,
            end_point,
        )

        s = gs.random.rand(1)
        points = gs.squeeze(geodesic(s), axis=-3)

        batch_shape = (n_points,) if n_points > 1 else ()
        self.test_dist_with_endpoints(
            initial_point, end_point, points, gs.zeros(batch_shape), atol
        )
