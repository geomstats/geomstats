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
        """Check Christoffel symbols.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., dim, dim, dim]
            Expected Christoffel symbols.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_equation(self, tangent_vec, base_point, expected, atol=gs.atol):
        """Check geodesic equation.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., 2, dim]
            Expected velocity and acceleration.
        atol : float
            Absolute tolerance.
        """
        state = gs.stack([base_point, tangent_vec])

        res = self.space.metric.geodesic_equation(state)
        self.assertAllClose(res, expected, atol=atol)

    def test_exp(self, tangent_vec, base_point, expected, atol=gs.atol):
        """Check exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected point on the manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.exp(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_log(self, point, base_point, expected, atol=gs.atol):
        """Check logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.log(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_riemann_tensor(self, base_point, expected, atol=gs.atol):
        """Check Riemann tensor.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., dim, dim, dim, dim]
            Expected Riemann tensor.
        atol : float
            Absolute tolerance.
        """
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
        """Check curvature.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.curvature(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_ricci_tensor(self, base_point, expected, atol=gs.atol):
        """Check Ricci tensor.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., dim, dim]
            Expected Ricci tensor.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.ricci_tensor(base_point)
        self.assertAllClose(res, expected, atol=atol)

    def test_directional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
    ):
        """Check directional curvature.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
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
        """Check covariant derivative of the curvature.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_d : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.curvature_derivative(
            tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d, base_point
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_directional_curvature_derivative(
        self, tangent_vec_a, tangent_vec_b, base_point, expected, atol=gs.atol
    ):
        """Check covariant derivative of the directional curvature.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected tangent vector at `base_point`.
        atol : float
            Absolute tolerance.
        """
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
        """Check geodesic.

        The geodesic is defined by `end_point` (boundary value problem)
        or by `initial_tangent_vec` (initial value problem).

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        expected : array-like, shape=[..., n_times, *space.shape]
            Expected points on the geodesic.
        atol : float
            Absolute tolerance.
        end_point : array-like, shape=[..., *space.shape]
            End point of the geodesic.
            Optional, default: None.
        initial_tangent_vec : array-like, shape=[..., *space.shape]
            Initial velocity of the geodesic.
            Optional, default: None.
        """
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
        """Check geodesic defined by the initial value problem.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        initial_tangent_vec : array-like, shape=[..., *space.shape]
            Initial velocity of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        expected : array-like, shape=[..., n_times, *space.shape]
            Expected points on the geodesic.
        atol : float
            Absolute tolerance.
        """
        self.test_geodesic(
            initial_point,
            time,
            expected,
            atol=atol,
            initial_tangent_vec=initial_tangent_vec,
        )

    def test_parallel_transport(
        self,
        tangent_vec,
        base_point,
        expected,
        atol=gs.atol,
        direction=None,
        end_point=None,
    ):
        """Check parallel transport.

        The geodesic along which to transport is defined by `direction`
        (initial value problem) or by `end_point` (boundary value problem).

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        expected : array-like, shape=[..., *space.shape]
            Expected transported tangent vector.
        atol : float
            Absolute tolerance.
        direction : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` along which to transport.
            Optional, default: None.
        end_point : array-like, shape=[..., *space.shape]
            Point on the manifold to transport to.
            Optional, default: None.
        """
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
        """Check parallel transport along a direction.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        direction : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` along which to transport.
        expected : array-like, shape=[..., *space.shape]
            Expected transported tangent vector.
        atol : float
            Absolute tolerance.
        """
        self.test_parallel_transport(
            tangent_vec, base_point, expected, atol, direction=direction
        )

    def test_parallel_transport_bvp(
        self, tangent_vec, base_point, end_point, expected, atol=gs.atol
    ):
        """Check parallel transport to an end point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        end_point : array-like, shape=[..., *space.shape]
            Point on the manifold to transport to.
        expected : array-like, shape=[..., *space.shape]
            Expected transported tangent vector.
        atol : float
            Absolute tolerance.
        """
        self.test_parallel_transport(
            tangent_vec, base_point, expected, atol, end_point=end_point
        )

    def test_exp_belongs(self, tangent_vec, base_point, atol=gs.atol):
        """Check exponential gives a point on the manifold.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        point = self.space.metric.exp(tangent_vec, base_point)

        assert_belongs(self.space, point, atol=atol)

    def test_log_is_tangent(self, point, base_point, atol=gs.atol):
        """Check logarithm gives a tangent vector.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.space.metric.log(point, base_point)

        assert_is_tangent(self.space, tangent_vec, base_point, atol=atol)

    def test_exp_after_log(self, point, base_point, atol=gs.atol):
        """Check exponential is a left inverse of the logarithm.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        tangent_vec = self.space.metric.log(point, base_point)
        point_ = self.space.metric.exp(tangent_vec, base_point)

        self.assertAllClose(point_, point, atol=atol)

    def test_log_after_exp(self, tangent_vec, base_point, atol=gs.atol):
        """Check logarithm is a left inverse of the exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        end_point = self.space.metric.exp(tangent_vec, base_point)
        tangent_vec_ = self.space.metric.log(end_point, base_point)

        self.assertAllClose(tangent_vec_, tangent_vec, atol=atol)

    def test_geodesic_ivp_belongs(
        self, initial_point, initial_tangent_vec, time, atol=gs.atol
    ):
        """Check geodesic belongs to the manifold.

        The geodesic is defined by the initial value problem.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        initial_tangent_vec : array-like, shape=[..., *space.shape]
            Initial velocity of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
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

        The geodesic starts at `base_point` with velocity `tangent_vec`
        and is evaluated at time 1.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(
            base_point, initial_tangent_vec=tangent_vec
        )

        end_point = self.space.metric.exp(tangent_vec, base_point)
        end_point_ = gs.squeeze(geod_func(1.0), axis=-(self.space.point_ndim + 1))

        self.assertAllClose(end_point_, end_point, atol=atol)

    def test_parallel_transport_ivp_transported_is_tangent(
        self, tangent_vec, base_point, direction, atol=gs.atol
    ):
        """Check parallel transport along a direction gives a tangent vector.

        The transported vector is tangent at the exponential of `direction`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        direction : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` along which to transport.
        atol : float
            Absolute tolerance.
        """
        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, direction=direction
        )

        end_point = self.space.metric.exp(direction, base_point)

        assert_is_tangent(self.space, transported, end_point, atol=atol)

    def test_parallel_transport_bvp_transported_is_tangent(
        self, tangent_vec, base_point, end_point, atol=gs.atol
    ):
        """Check parallel transport to an end point gives a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        end_point : array-like, shape=[..., *space.shape]
            Point on the manifold to transport to.
        atol : float
            Absolute tolerance.
        """
        transported = self.space.metric.parallel_transport(
            tangent_vec, base_point, end_point=end_point
        )

        assert_is_tangent(self.space, transported, end_point, atol=atol)


class ConnectionComparisonTestCase(TestCase):
    """Test case comparing the connections of two spaces.

    Each test checks a method gives the same result for `space` and
    `other_space`.
    """

    def test_christoffels(self, base_point, atol=gs.atol):
        """Check Christoffel symbols is the same for both spaces.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.christoffels(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_exp(self, tangent_vec, base_point, atol=gs.atol):
        """Check exponential is the same for both spaces.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.exp(tangent_vec, base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_log(self, point, base_point, atol=gs.atol):
        """Check logarithm is the same for both spaces.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.log(point, base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_riemann_tensor(self, base_point, atol=gs.atol):
        """Check Riemann tensor is the same for both spaces.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.riemann_tensor(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_curvature(
        self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point, atol=gs.atol
    ):
        """Check curvature is the same for both spaces.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.curvature(
                tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_ricci_tensor(self, base_point, atol=gs.atol):
        """Check Ricci tensor is the same for both spaces.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.ricci_tensor(base_point),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_directional_curvature(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check directional curvature is the same for both spaces.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
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
        """Check covariant derivative of the curvature is the same for both spaces.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_d : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
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
        """Check directional curvature derivative is the same for both spaces.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.directional_curvature_derivative(
                tangent_vec_a, tangent_vec_b, base_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_geodesic_bvp(self, initial_point, end_point, time, atol=gs.atol):
        """Check geodesic is the same for both spaces.

        The geodesic is defined by the boundary value problem.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        end_point : array-like, shape=[..., *space.shape]
            End point of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.geodesic(initial_point, end_point=end_point)(
                time
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_geodesic_ivp(self, initial_point, initial_tangent_vec, time, atol=gs.atol):
        """Check geodesic is the same for both spaces.

        The geodesic is defined by the initial value problem.

        Parameters
        ----------
        initial_point : array-like, shape=[..., *space.shape]
            Initial point of the geodesic.
        initial_tangent_vec : array-like, shape=[..., *space.shape]
            Initial velocity of the geodesic.
        time : array-like, shape=[n_times,]
            Times at which to evaluate the geodesic.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.geodesic(
                initial_point, initial_tangent_vec=initial_tangent_vec
            )(time),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_parallel_transport_ivp(
        self, tangent_vec, base_point, direction, atol=gs.atol
    ):
        """Check parallel transport along a direction is the same for both spaces.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        direction : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` along which to transport.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.parallel_transport(
                tangent_vec, base_point, direction=direction
            ),
            self.space,
            self.other_space,
            atol=atol,
        )

    def test_parallel_transport_bvp(
        self, tangent_vec, base_point, end_point, atol=gs.atol
    ):
        """Check parallel transport to an end point is the same for both spaces.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., *space.shape]
            Tangent vector at `base_point` to transport.
        base_point : array-like, shape=[..., *space.shape]
            Point on the manifold.
        end_point : array-like, shape=[..., *space.shape]
            Point on the manifold to transport to.
        atol : float
            Absolute tolerance.
        """
        assert_same_result(
            lambda space: space.metric.parallel_transport(
                tangent_vec, base_point, end_point=end_point
            ),
            self.space,
            self.other_space,
            atol=atol,
        )
