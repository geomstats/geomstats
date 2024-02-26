import math

import pytest

import geomstats.backend as gs
from geomstats.test.random import get_random_times
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.vectorization import get_batch_shape


class ProjectionTestCaseMixins:
    def test_projection(self, point, expected, atol):
        proj_point = self.space.projection(point)
        self.assertAllClose(proj_point, expected, atol=atol)

    @pytest.mark.vec
    def test_projection_vec(self, n_reps, atol):
        point = self.data_generator.point_to_project()
        expected = self.space.projection(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_projection_belongs(self, n_points, atol):
        """Check projection belongs to manifold.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point = self.data_generator.point_to_project(n_points)
        proj_point = self.space.projection(point)
        expected = gs.ones(n_points, dtype=bool)

        self.test_belongs(proj_point, expected, atol)


class GroupExpTestCaseMixins:
    def test_exp(self, tangent_vec, base_point, expected, atol):
        point = self.space.exp(tangent_vec, base_point)
        self.assertAllClose(point, expected, atol=atol)


class DistTestCaseMixins:
    def test_dist(self, point_a, point_b, expected, atol):
        res = self.space.metric.dist(point_a, point_b)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_dist_is_symmetric(self, n_points, atol):
        """Check distance is symmetric.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        dist_ab = self.space.metric.dist(point_a, point_b)
        dist_ba = self.space.metric.dist(point_b, point_a)

        self.assertAllClose(dist_ab, dist_ba, atol=atol)

    @pytest.mark.random
    def test_dist_is_positive(self, n_points, atol):
        """Check distance is positive.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        dist_ = self.space.metric.dist(point_a, point_b)
        res = gs.all(dist_ > -atol)
        self.assertTrue(res)

    @pytest.mark.random
    def test_dist_point_to_itself_is_zero(self, n_points, atol):
        """Check distance of a point to itself is zero.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point = self.data_generator.random_point(n_points)

        dist_ = self.space.metric.dist(point, point)

        expected_shape = get_batch_shape(self.space.point_ndim, point)
        expected = gs.zeros(expected_shape)
        self.assertAllClose(dist_, expected, atol=atol)

    @pytest.mark.random
    def test_dist_triangle_inequality(self, n_points, atol):
        """Check distance satifies triangle inequality.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)
        point_c = self.data_generator.random_point(n_points)

        dist_ab = self.space.metric.dist(point_a, point_b)
        dist_bc = self.space.metric.dist(point_b, point_c)
        rhs = dist_ac = self.space.metric.dist(point_a, point_c)

        lhs = dist_ab + dist_bc
        res = gs.all(lhs + atol >= rhs)
        self.assertTrue(res, f"lhs: {lhs}, rhs: {dist_ac}, diff: {lhs-rhs}")


class GeodesicBVPTestCaseMixins:
    @pytest.mark.vec
    def test_geodesic_bvp_vec(self, n_reps, n_times, atol):
        initial_point, end_point = self.data_generator.random_point(2)
        time = get_random_times(n_times)

        expected = self.space.metric.geodesic(initial_point, end_point=end_point)(time)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    initial_point=initial_point,
                    end_point=end_point,
                    time=time,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["initial_point", "end_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data, test_fnc_name="test_geodesic")

    @pytest.mark.random
    def test_geodesic_boundary_points(self, n_points, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        time = gs.array([0.0, 1.0])

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)

        res = geod_func(time)
        expected = gs.stack(
            [initial_point, end_point], axis=-(self.space.point_ndim + 1)
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

        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_bvp_belongs(self, n_points, n_times, atol):
        """Check geodesic belongs to manifold.

        This is for geodesics defined by the boundary value problem (bvp).

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        time = get_random_times(n_times)

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        points = geod_func(time)

        res = self.space.belongs(gs.reshape(points, (-1, *self.space.shape)), atol=atol)

        expected_shape = (
            math.prod(get_batch_shape(self.space.point_ndim, initial_point)) * n_times,
        )
        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)
