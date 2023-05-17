import math

import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator, get_random_times
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.vectorization import get_batch_shape


class ConnectionTestCase(TestCase):
    # TODO: geodesic and inverse parametrization geodesic
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_christoffels(self, base_point, expected, atol):
        res = self.space.metric.christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_christoffels_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.christoffels(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_exp(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_exp_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.exp(tangent_vec, base_point)

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
            vectorization_type="repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_exp_belongs(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        point = self.space.metric.exp(tangent_vec, base_point)

        res = self.space.belongs(point, atol=atol)
        expected_shape = get_batch_shape(self.space, base_point)
        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_log(self, point, base_point, expected, atol):
        res = self.space.metric.log(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_log_vec(self, n_reps, atol):
        point, base_point = self.data_generator.random_point(2)

        expected = self.space.metric.log(point, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_log_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        point = self.data_generator.random_point(n_points)

        tangent_vec = self.space.metric.log(point, base_point)

        res = self.space.is_tangent(tangent_vec, base_point)
        expected_shape = get_batch_shape(self.space, base_point)
        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_exp_after_log(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        tangent_vec = self.space.metric.log(end_point, base_point)
        end_point_ = self.space.metric.exp(tangent_vec, base_point)

        self.assertAllClose(end_point_, end_point, atol=atol)

    @pytest.mark.random
    def test_log_after_exp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        end_point = self.space.metric.exp(tangent_vec, base_point)
        tangent_vec_ = self.space.metric.log(end_point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_riemann_tensor(self, base_point, expected, atol):
        res = self.space.metric.riemann_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_riemann_tensor_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.riemann_tensor(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_curvature(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point, expected, atol
    ):
        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_curvature_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    tangent_vec_c=tangent_vec_c,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "tangent_vec_c", "base_point"],
            expected_name="expected",
            vectorization_type="repeat-0-1-2",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_ricci_tensor(self, base_point, expected, atol):
        res = self.space.metric.ricci_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_ricci_tensor_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.ricci_tensor(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_directional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_directional_curvature_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            vectorization_type="repeat-0-1",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point,
        expected,
        atol,
    ):
        res = self.space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_curvature_derivative_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_d = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    tangent_vec_c=tangent_vec_c,
                    tangent_vec_d=tangent_vec_d,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=[
                "tangent_vec_a",
                "tangent_vec_b",
                "tangent_vec_c",
                "tangent_vec_d",
                "base_point",
            ],
            expected_name="expected",
            vectorization_type="repeat-0-1-2-3",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_directional_curvature_derivative_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec_a", "tangent_vec_b", "base_point"],
            expected_name="expected",
            vectorization_type="repeat-0-1",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_geodesic(
        self,
        initial_point,
        time,
        expected,
        atol,
        end_point=None,
        initial_tangent_vec=None,
    ):
        geod_func = self.space.metric.geodesic(
            initial_point, end_point=end_point, initial_tangent_vec=initial_tangent_vec
        )
        res = geod_func(time)
        self.assertAllClose(res, expected, atol=atol)

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

    @pytest.mark.vec
    def test_geodesic_ivp_vec(self, n_reps, n_times, atol):
        initial_point = self.data_generator.random_point()
        initial_tangent_vec = self.data_generator.random_tangent_vec(initial_point)
        time = get_random_times(n_times)

        expected = self.space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )(time)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    initial_point=initial_point,
                    initial_tangent_vec=initial_tangent_vec,
                    time=time,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["initial_point", "initial_tangent_vec"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="repeat-1",
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
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        time = get_random_times(n_times)

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        points = geod_func(time)

        res = self.space.belongs(gs.reshape(points, (-1, *self.space.shape)))

        expected_shape = (
            math.prod(get_batch_shape(self.space, initial_point)) * n_times,
        )
        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_geodesic_ivp_belongs(self, n_points, n_times, atol):
        initial_point = self.data_generator.random_point(n_points)
        initial_tangent_vec = self.data_generator.random_tangent_vec(initial_point)

        time = get_random_times(n_times)

        geod_func = self.space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )

        points = geod_func(time)

        res = self.space.belongs(gs.reshape(points, (-1, *self.space.shape)))

        expected_shape = (
            math.prod(get_batch_shape(self.space, initial_point)) * n_times,
        )
        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_exp_geodesic_ivp(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        geod_func = self.space.metric.geodesic(
            base_point, initial_tangent_vec=tangent_vec
        )

        end_point = self.space.metric.exp(tangent_vec, base_point)
        end_point_ = gs.squeeze(geod_func(1.0), axis=-(self.space.point_ndim + 1))

        self.assertAllClose(end_point_, end_point, atol=atol)

    def test_parallel_transport(
        self, tangent_vec, base_point, expected, atol, direction=None, end_point=None
    ):
        res = self.space.metric.parallel_transport(
            tangent_vec,
            base_point,
            direction=direction,
            end_point=end_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_parallel_transport_vec_with_direction(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        direction = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    direction=direction,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point", "direction"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="repeat-0-2",
        )
        self._test_vectorization(vec_data, test_fnc_name="test_parallel_transport")

    @pytest.mark.vec
    def test_parallel_transport_vec_with_end_point(self, n_reps, atol):
        base_point, end_point = self.data_generator.random_point(2)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    end_point=end_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point", "end_point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="repeat-0-1",
        )
        self._test_vectorization(vec_data, test_fnc_name="test_parallel_transport")

    @pytest.mark.random
    def test_parallel_transport_transported_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        res = self.space.is_tangent(transported, end_point, atol=atol)

        expected_shape = get_batch_shape(self.space, base_point)
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(res, expected)

    def test_injectivity_radius(self, base_point, expected, atol):
        res = self.space.metric.injectivity_radius(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_injectivity_radius_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.injectivity_radius(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
