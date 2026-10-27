import geomstats.backend as gs
from geomstats.test.assertions import (
    assert_belongs,
    assert_is_tangent,
    assert_same_result,
)
from geomstats.test.test_case import TestCase

from .mixins import GeodesicBVPTestCaseMixins


class ConnectionTestCase(GeodesicBVPTestCaseMixins, TestCase):
    def test_christoffels(self, base_point, expected, atol=gs.atol):
        res = self.space.metric.christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_equation(self, tangent_vec, base_point, expected, atol=gs.atol):
        state = gs.stack([base_point, tangent_vec])

        res = self.space.metric.geodesic_equation(state)
        self.assertAllClose(res, expected, atol=atol)

    def test_exp(self, tangent_vec, base_point, expected, atol=gs.atol):
        res = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_exp_belongs(self, tangent_vec, base_point, atol=gs.atol):
        """Check exponential gives point in the manifold.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        point = self.space.metric.exp(tangent_vec, base_point)

        assert_belongs(self.space, point, atol=atol)

    def test_log(self, point, base_point, expected, atol=gs.atol):
        res = self.space.metric.log(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_log_is_tangent(self, point, base_point, atol=gs.atol):
        """Check logarithm gives a tangent vector.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.space.metric.log(point, base_point)

        assert_is_tangent(self.space, tangent_vec, base_point, atol=atol)

    def test_exp_after_log(self, point, base_point, atol=gs.atol):
        """Check exp and log are inverse.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.space.metric.log(point, base_point)
        point_ = self.space.metric.exp(tangent_vec, base_point)

        self.assertAllClose(point_, point, atol=atol)

    def test_log_after_exp(self, tangent_vec, base_point, atol=gs.atol):
        """Check log and exp are inverse.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        end_point = self.space.metric.exp(tangent_vec, base_point)
        tangent_vec_ = self.space.metric.log(end_point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_riemann_tensor(self, base_point, expected, atol=gs.atol):
        res = self.space.metric.riemann_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_curvature(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        base_point,
        expected,
        atol=gs.atol,
    ):
        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_ricci_tensor(self, base_point, expected, atol=gs.atol):
        res = self.space.metric.ricci_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_directional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
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
        atol=gs.atol,
    ):
        res = self.space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
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
        atol=gs.atol,
        end_point=None,
        initial_tangent_vec=None,
    ):
        geod_func = self.space.metric.geodesic(
            initial_point, end_point=end_point, initial_tangent_vec=initial_tangent_vec
        )
        res = geod_func(time)
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_ivp(
        self,
        initial_point,
        initial_tangent_vec,
        time,
        expected,
        atol=gs.atol,
    ):
        self.test_geodesic(
            initial_point,
            time,
            expected,
            atol=atol,
            initial_tangent_vec=initial_tangent_vec,
        )

    def test_geodesic_ivp_belongs(
        self, initial_point, initial_tangent_vec, time, atol=gs.atol
    ):
        """Check geodesic belongs to manifold.

        This is for geodesics defined by the initial value problem (ivp).

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )

        points = geod_func(time)

        assert_belongs(self.space, points, atol=atol)

    def test_exp_geodesic_ivp(self, tangent_vec, base_point, atol=gs.atol):
        """Check end point of a geodesic matches exponential.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(
            base_point, initial_tangent_vec=tangent_vec
        )

        end_point = self.space.metric.exp(tangent_vec, base_point)
        end_point_ = gs.squeeze(geod_func(1.0), axis=-(self.space.point_ndim + 1))

        self.assertAllClose(end_point_, end_point, atol=atol)

    def test_parallel_transport(
        self,
        tangent_vec,
        base_point,
        expected,
        atol=gs.atol,
        direction=None,
        end_point=None,
    ):
        res = self.space.metric.parallel_transport(
            tangent_vec,
            base_point,
            direction=direction,
            end_point=end_point,
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_parallel_transport_ivp(
        self, tangent_vec, base_point, direction, expected, atol=gs.atol
    ):
        self.test_parallel_transport(
            tangent_vec, base_point, expected, atol, direction=direction
        )

    def test_parallel_transport_bvp(
        self, tangent_vec, base_point, end_point, expected, atol=gs.atol
    ):
        self.test_parallel_transport(
            tangent_vec, base_point, expected, atol, end_point=end_point
        )

    def test_parallel_transport_ivp_transported_is_tangent(
        self, tangent_vec, base_point, direction, atol=gs.atol
    ):
        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )

        end_point = self.space.metric.exp(direction, base_point)

        assert_is_tangent(self.space, transported, end_point, atol=atol)

    def test_parallel_transport_bvp_transported_is_tangent(
        self, tangent_vec, base_point, end_point, atol=gs.atol
    ):
        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        assert_is_tangent(self.space, transported, end_point, atol=atol)


class ConnectionComparisonTestCase(TestCase):
    def test_christoffels(self, base_point, atol=gs.atol):
        assert_same_result(
            lambda space: space.metric.christoffels(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_exp(self, tangent_vec, base_point, atol=gs.atol):
        assert_same_result(
            lambda space: space.metric.exp(tangent_vec, base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_log(self, point, base_point, atol=gs.atol):
        assert_same_result(
            lambda space: space.metric.log(point, base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_riemann_tensor(self, base_point, atol=gs.atol):
        assert_same_result(
            lambda space: space.metric.riemann_tensor(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_curvature(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point, atol=gs.atol
    ):
        assert_same_result(
            lambda space: space.metric.curvature(
                tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_ricci_tensor(self, base_point, atol=gs.atol):
        assert_same_result(
            lambda space: space.metric.ricci_tensor(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_directional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        assert_same_result(
            lambda space: space.metric.directional_curvature(
                tangent_vec_a, tangent_vec_b, base_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point,
        atol=gs.atol,
    ):
        assert_same_result(
            lambda space: space.metric.curvature_derivative(
                tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        assert_same_result(
            lambda space: space.metric.directional_curvature_derivative(
                tangent_vec_a, tangent_vec_b, base_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_geodesic_bvp(self, initial_point, end_point, time, atol=gs.atol):
        assert_same_result(
            lambda space: space.metric.geodesic(initial_point, end_point=end_point)(
                time
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_geodesic_ivp(self, initial_point, initial_tangent_vec, time, atol=gs.atol):
        assert_same_result(
            lambda space: space.metric.geodesic(
                initial_point, initial_tangent_vec=initial_tangent_vec
            )(time),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_parallel_transport_ivp(
        self, base_point, tangent_vec, direction, atol=gs.atol
    ):
        assert_same_result(
            lambda space: space.metric.parallel_transport(
                tangent_vec, base_point, direction=direction
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_parallel_transport_bvp(
        self, base_point, end_point, tangent_vec, atol=gs.atol
    ):
        assert_same_result(
            lambda space: space.metric.parallel_transport(
                tangent_vec, base_point, end_point=end_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )
