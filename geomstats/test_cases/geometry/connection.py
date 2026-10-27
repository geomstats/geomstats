import geomstats.backend as gs
from geomstats.test.test_case import TestCase
from geomstats.vectorization import get_batch_shape

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

        res = self.space.belongs(point, atol=atol)
        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

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

        res = self.space.is_tangent(tangent_vec, base_point, atol=atol)
        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

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
        res = self.space.belongs(points, atol=atol)

        batch_shape = get_batch_shape(
            self.space.point_ndim,
            initial_point,
            initial_tangent_vec,
        )
        expected_shape = batch_shape + (time.shape or (1,))

        expected = gs.ones(expected_shape, dtype=bool)
        self.assertAllEqual(res, expected)

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

        res = self.space.is_tangent(transported, end_point, atol=atol)

        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(res, expected)

    def test_parallel_transport_bvp_transported_is_tangent(
        self, tangent_vec, base_point, end_point, atol=gs.atol
    ):
        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        res = self.space.is_tangent(transported, end_point, atol=atol)

        expected_shape = get_batch_shape(self.space.point_ndim, base_point)
        expected = gs.ones(expected_shape, dtype=bool)

        self.assertAllEqual(res, expected)


class ConnectionComparisonTestCase(TestCase):
    def test_christoffels(self, base_point, atol=gs.atol):
        res = self.space.metric.christoffels(base_point)
        res_ = self.other_space.metric.christoffels(base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_exp(self, tangent_vec, base_point, atol=gs.atol):
        res = self.space.metric.exp(tangent_vec, base_point)
        res_ = self.other_space.metric.exp(tangent_vec, base_point)

        self.assertAllClose(res, res_, atol=atol)

    def test_log(self, point, base_point, atol=gs.atol):
        res = self.space.metric.log(point, base_point)
        res_ = self.other_space.metric.log(point, base_point)

        self.assertAllClose(res, res_, atol=atol)

    def test_riemann_tensor(self, base_point, atol=gs.atol):
        res = self.space.metric.riemann_tensor(base_point)
        res_ = self.other_space.metric.riemann_tensor(base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_curvature(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point, atol=gs.atol
    ):
        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        res_ = self.other_space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    def test_ricci_tensor(self, base_point, atol=gs.atol):
        res = self.space.metric.ricci_tensor(base_point)
        res_ = self.other_space.metric.ricci_tensor(base_point)
        self.assertAllClose(res, res_, atol=atol)

    def test_directional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        res = self.space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        res_ = self.other_space.metric.directional_curvature(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    def test_curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b,
        tangent_vec_c,
        tangent_vec_d,
        base_point,
        atol=gs.atol,
    ):
        res = self.space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        res_ = self.other_space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    def test_directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        res = self.space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )
        res_ = self.other_space.metric.directional_curvature_derivative(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, res_, atol=atol)

    def test_geodesic_bvp(self, initial_point, end_point, time, atol=gs.atol):
        res = self.space.metric.geodesic(initial_point, end_point=end_point)(time)
        res_ = self.other_space.metric.geodesic(initial_point, end_point=end_point)(
            time
        )

        self.assertAllClose(res, res_, atol=atol)

    def test_geodesic_ivp(self, initial_point, initial_tangent_vec, time, atol=gs.atol):
        res = self.space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )(time)

        res_ = self.other_space.metric.geodesic(
            initial_point, initial_tangent_vec=initial_tangent_vec
        )(time)

        self.assertAllClose(res, res_, atol=atol)

    def test_parallel_transport_ivp(
        self, base_point, tangent_vec, direction, atol=gs.atol
    ):
        res = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )
        res_ = self.other_space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )
        self.assertAllClose(res, res_, atol=atol)

    def test_parallel_transport_bvp(
        self, base_point, end_point, tangent_vec, atol=gs.atol
    ):
        res = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )
        res_ = self.other_space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        self.assertAllClose(res, res_, atol=atol)
