import math

import pytest

import geomstats.backend as gs
from geomstats.test.random import RandomDataGenerator, get_random_times
from geomstats.test.test_case import TestCase
from geomstats.test.utils import IdentityPointTransformer, PointTransformerFromDiffeo
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.mixins import GeodesicBVPTestCaseMixins
from geomstats.vectorization import get_batch_shape


class ConnectionTestCase(GeodesicBVPTestCaseMixins, TestCase):
    tangent_to_multiple = False
    is_metric = True

    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

    def test_christoffels(self, base_point, expected, atol):
        res = self.space.metric.christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_equation(self, state, expected, atol):
        res = self.space.metric.geodesic_equation(state)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_geodesic_equation_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        state = gs.stack([base_point, tangent_vec])
        expected = self.space.metric.geodesic_equation(state)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    state=state,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["state"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_exp(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_exp_belongs(self, n_points, atol):
        """Check exponential gives point in the manifold.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        point = self.space.metric.exp(tangent_vec, base_point)

        res = self.space.belongs(point, atol=atol)
        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

    def test_log(self, point, base_point, expected, atol):
        res = self.space.metric.log(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_log_is_tangent(self, n_points, atol):
        """Check logarithm gives a tangent vector.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        base_point = self.data_generator.random_point(n_points)
        point = self.data_generator.random_point(n_points)

        tangent_vec = self.space.metric.log(point, base_point)

        res = self.space.is_tangent(tangent_vec, base_point, atol=atol)
        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_exp_after_log(self, n_points, atol):
        """Check exp and log are inverse.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)

        tangent_vec = self.space.metric.log(end_point, base_point)
        end_point_ = self.space.metric.exp(tangent_vec, base_point)

        self.assertAllClose(end_point_, end_point, atol=atol)

    @pytest.mark.random
    def test_log_after_exp(self, n_points, atol):
        """Check log and exp are inverse.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        end_point = self.space.metric.exp(tangent_vec, base_point)
        tangent_vec_ = self.space.metric.log(end_point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_riemann_tensor(self, base_point, expected, atol):
        res = self.space.metric.riemann_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_curvature(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point, expected, atol
    ):
        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_ricci_tensor(self, base_point, expected, atol):
        res = self.space.metric.ricci_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_directional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

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

    def test_directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    ):
        res = self.space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

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
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-1",
        )
        self._test_vectorization(vec_data, test_fnc_name="test_geodesic")

    @pytest.mark.random
    def test_geodesic_ivp_belongs(self, n_points, n_times, atol):
        """Check geodesic belongs to manifold.

        This is for geodesics defined by the initial value problem (ivp).

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
        initial_point = self.data_generator.random_point(n_points)
        initial_tangent_vec = self.data_generator.random_tangent_vec(initial_point)

        time = get_random_times(n_times)

        geod_func = self.space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )

        points = geod_func(time)

        res = self.space.belongs(gs.reshape(points, (-1, *self.space.shape)), atol=atol)

        expected_shape = (
            math.prod(get_batch_shape(self.space.point_ndim, initial_point)) * n_times,
        )

        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_exp_geodesic_ivp(self, n_points, atol):
        """Check end point of a geodesic matches exponential.

        Parameters
        ----------
        n_points : int
            Number of random points to generate.
        atol : float
            Absolute tolerance.
        """
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
    def test_parallel_transport_ivp_vec(self, n_reps, atol):
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
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0-2",
        )
        self._test_vectorization(vec_data, test_fnc_name="test_parallel_transport")

    @pytest.mark.vec
    def test_parallel_transport_bvp_vec(self, n_reps, atol):
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
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
        )
        self._test_vectorization(vec_data, test_fnc_name="test_parallel_transport")

    @pytest.mark.random
    def test_parallel_transport_bvp_transported_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        res = self.space.is_tangent(transported, end_point, atol=atol)

        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(res, expected)

    @pytest.mark.random
    def test_parallel_transport_ivp_transported_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        direction = self.data_generator.random_tangent_vec(base_point)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )

        end_point = self.space.metric.exp(direction, base_point)

        res = self.space.is_tangent(transported, end_point, atol=atol)

        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(res, expected)

    def test_injectivity_radius(self, base_point, expected, atol):
        res = self.space.metric.injectivity_radius(base_point)
        self.assertAllClose(res, expected, atol=atol)


class ConnectionComparisonTestCase(TestCase):
    def setup_method(self):
        if not hasattr(self, "data_generator"):
            self.data_generator = RandomDataGenerator(self.space)

        if not hasattr(self, "point_transformer") and hasattr(self, "diffeo"):
            self.point_transformer = PointTransformerFromDiffeo(self.diffeo)

        if not hasattr(self, "point_transformer"):
            self.point_transformer = IdentityPointTransformer()

    def test_christoffels(self, base_point, atol):
        res = self.space.metric.christoffels(base_point)
        res_ = self.other_space.metric.christoffels(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_christoffels_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        self.test_christoffels(base_point, atol)

    def test_exp(self, tangent_vec, base_point, atol):
        base_point_ = self.point_transformer.transform_point(base_point)
        tangent_vec_ = self.point_transformer.transform_tangent_vec(
            tangent_vec, base_point
        )

        res = self.space.metric.exp(tangent_vec, base_point)
        res_ = self.other_space.metric.exp(tangent_vec_, base_point_)

        res_ = self.point_transformer.inverse_transform_point(res_)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_exp_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        self.test_exp(tangent_vec, base_point, atol)

    def test_log(self, point, base_point, atol):
        base_point_ = self.point_transformer.transform_point(base_point)
        point_ = self.point_transformer.transform_point(point)

        res = self.space.metric.log(point, base_point)
        res_ = self.other_space.metric.log(point_, base_point_)

        res_ = self.point_transformer.inverse_transform_tangent_vec(res_, base_point_)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_log_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        point = self.data_generator.random_point(n_points)

        self.test_log(point, base_point, atol)

    def test_riemann_tensor(self, base_point, atol):
        base_point_ = self.point_transformer.transform_point(base_point)

        res = self.space.metric.riemann_tensor(base_point)
        res_ = self.other_space.metric.riemann_tensor(base_point_)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_riemann_tensor_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        self.test_riemann_tensor(base_point, atol)

    def test_curvature(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point, atol
    ):
        base_point_ = self.point_transformer.transform_point(base_point)
        tangent_vec_a_ = self.point_transformer.transform_tangent_vec(
            tangent_vec_a, base_point
        )
        tangent_vec_b_ = self.point_transformer.transform_tangent_vec(
            tangent_vec_b, base_point
        )
        tangent_vec_c_ = self.point_transformer.transform_tangent_vec(
            tangent_vec_c, base_point
        )

        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        res_ = self.other_space.metric.curvature(
            tangent_vec_a_, tangent_vec_b_, tangent_vec_c_, base_point_
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_curvature_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)

        self.test_curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point, atol
        )

    def test_ricci_tensor(self, base_point, atol):
        res = self.space.metric.ricci_tensor(base_point)
        res_ = self.other_space.metric.ricci_tensor(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_ricci_tensor_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        self.test_ricci_tensor(base_point, atol)

    def test_directional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, atol
    ):
        res = self.space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        res_ = self.other_space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_directional_curvature_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        self.test_directional_curvature(tangent_vec_a, tangent_vec_b, base_point, atol)

    def test_curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point,
        atol,
    ):
        res = self.space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        res_ = self.other_space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_curvature_derivative_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_c = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_d = self.data_generator.random_tangent_vec(base_point)

        self.test_curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point, atol
        )

    def test_directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point, atol
    ):
        res = self.space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )
        res_ = self.other_space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_directional_curvature_derivative_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        self.test_directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point, atol
        )

    def test_geodesic_bvp(self, initial_point, end_point, time, atol):
        initial_point_ = self.point_transformer.transform_point(initial_point)
        end_point_ = self.point_transformer.transform_point(end_point)

        res = self.space.metric.geodesic(initial_point, end_point=end_point)(time)
        res_ = self.other_space.metric.geodesic(initial_point_, end_point=end_point_)(
            time
        )

        res_ = self.point_transformer.inverse_transform_point(res_)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_bvp_random(self, n_points, n_times, atol):
        initial_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        time = get_random_times(n_times)

        self.test_geodesic_bvp(initial_point, end_point, time, atol)

    def test_geodesic_ivp(self, initial_point, initial_tangent_vec, time, atol):
        initial_point_ = self.point_transformer.transform_point(initial_point)
        initial_tangent_vec_ = self.point_transformer.transform_tangent_vec(
            initial_tangent_vec, initial_point
        )

        res = self.space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )(time)

        res_ = self.other_space.metric.geodesic(
            initial_point_, initial_tangent_vec=initial_tangent_vec_
        )(time)

        res_ = self.point_transformer.inverse_transform_point(res_)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_geodesic_ivp_random(self, n_points, n_times, atol):
        initial_point = self.data_generator.random_point(n_points)
        initial_tangent_vec = self.data_generator.random_tangent_vec(initial_point)
        time = get_random_times(n_times)

        self.test_geodesic_ivp(initial_point, initial_tangent_vec, time, atol)

    def test_parallel_transport_ivp(self, base_point, tangent_vec, direction, atol):
        base_point_ = self.point_transformer.transform_point(base_point)
        tangent_vec_ = self.point_transformer.transform_tangent_vec(
            tangent_vec, base_point
        )
        direction_ = self.point_transformer.transform_tangent_vec(direction, base_point)

        res = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )
        res_ = self.other_space.metric.parallel_transport(
            tangent_vec_, base_point_, direction=direction_
        )

        end_point_ = self.other_space.metric.exp(direction_, base_point_)
        res_ = self.point_transformer.inverse_transform_tangent_vec(res_, end_point_)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_parallel_transport_ivp_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)
        direction = self.data_generator.random_tangent_vec(base_point)

        self.test_parallel_transport_ivp(base_point, tangent_vec, direction, atol)

    def test_parallel_transport_bvp(self, base_point, end_point, tangent_vec, atol):
        base_point_ = self.point_transformer.transform_point(base_point)
        end_point_ = self.point_transformer.transform_point(end_point)
        tangent_vec_ = self.point_transformer.transform_tangent_vec(
            tangent_vec, base_point
        )

        res = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )
        res_ = self.other_space.metric.parallel_transport(
            tangent_vec_, base_point_, end_point=end_point_
        )

        res_ = self.point_transformer.inverse_transform_tangent_vec(res_, end_point_)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_parallel_transport_bvp_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        end_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        self.test_parallel_transport_bvp(base_point, end_point, tangent_vec, atol)

    def test_injectivity_radius(self, base_point, atol):
        res = self.space.metric.injectivity_radius(base_point)
        res_ = self.other_space.metric.injectivity_radius(base_point)
        self.assertAllClose(res, res_, atol=atol)

    @pytest.mark.random
    def test_injectivity_radius_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)

        self.test_injectivity_radius(base_point, atol)
