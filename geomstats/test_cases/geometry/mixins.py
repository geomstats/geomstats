import geomstats.backend as gs
from geomstats.test.assertions import assert_belongs
from geomstats.vectorization import get_batch_shape


class ProjectionTestCaseMixins:
    def test_projection(self, point, expected, atol=gs.atol):
        proj_point = self.space.projection(point)
        self.assertAllClose(proj_point, expected, atol=atol)

    def test_projection_belongs(self, point, atol=gs.atol):
        """Check projection belongs to manifold.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        proj_point = self.space.projection(point)

        assert_belongs(self.space, proj_point, atol=atol)


class GroupExpTestCaseMixins:
    def test_exp(self, tangent_vec, base_point, expected, atol):
        point = self.space.exp(tangent_vec, base_point)
        self.assertAllClose(point, expected, atol=atol)


class DistTestCaseMixins:
    def test_squared_dist(self, point_a, point_b, expected, atol=gs.atol):
        res = self.space.metric.squared_dist(point_a, point_b)
        self.assertAllClose(res, expected, atol=atol)

    def test_squared_dist_is_symmetric(self, point_a, point_b, atol=gs.atol):
        """Check squared distance is symmetric.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        squared_dist_ab = self.space.metric.squared_dist(point_a, point_b)
        squared_dist_ba = self.space.metric.squared_dist(point_b, point_a)

        self.assertAllClose(squared_dist_ab, squared_dist_ba, atol=atol)

    def test_squared_dist_is_positive(self, point_a, point_b, atol=gs.atol):
        """Check squared distance is positive.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        squared_dist_ = self.space.metric.squared_dist(point_a, point_b)
        res = gs.all(squared_dist_ > -atol)
        self.assertTrue(res)

    def test_dist(self, point_a, point_b, expected, atol=gs.atol):
        res = self.space.metric.dist(point_a, point_b)
        self.assertAllClose(res, expected, atol=atol)

    def test_dist_is_symmetric(self, point_a, point_b, atol=gs.atol):
        """Check distance is symmetric.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        dist_ab = self.space.metric.dist(point_a, point_b)
        dist_ba = self.space.metric.dist(point_b, point_a)

        self.assertAllClose(dist_ab, dist_ba, atol=atol)

    def test_dist_is_positive(self, point_a, point_b, atol=gs.atol):
        """Check distance is positive.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        dist_ = self.space.metric.dist(point_a, point_b)
        res = gs.all(dist_ > -atol)
        self.assertTrue(res)

    def test_dist_point_to_itself_is_zero(self, point, atol=gs.atol):
        """Check distance of a point to itself is zero.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        dist_ = self.space.metric.dist(point, point)

        batch_shape = get_batch_shape(self.space.point_ndim, point)
        expected = gs.zeros(batch_shape)
        self.assertAllClose(dist_, expected, atol=atol)

    def test_dist_triangle_inequality(self, point_a, point_b, point_c, atol=gs.atol):
        """Check distance satifies triangle inequality.

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        dist_ab = self.space.metric.dist(point_a, point_b)
        dist_bc = self.space.metric.dist(point_b, point_c)
        rhs = dist_ac = self.space.metric.dist(point_a, point_c)

        lhs = dist_ab + dist_bc
        res = gs.all(lhs + atol >= rhs)
        self.assertTrue(res, f"lhs: {lhs}, rhs: {dist_ac}, diff: {lhs - rhs}")


class GeodesicBVPTestCaseMixins:
    def test_geodesic_bvp(
        self,
        initial_point,
        end_point,
        time,
        expected,
        atol=gs.atol,
    ):
        self.test_geodesic(
            initial_point, time, expected, atol=atol, end_point=end_point
        )

    def test_geodesic_boundary_points(self, initial_point, end_point, atol=gs.atol):
        time = gs.array([0.0, 1.0])

        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)

        res = geod_func(time)
        expected = gs.stack(
            [initial_point, end_point], axis=-(self.space.point_ndim + 1)
        )
        self.assertAllClose(res, expected, atol=atol)

    def test_geodesic_bvp_reverse(self, initial_point, end_point, time, atol=gs.atol):
        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        geod_func_reverse = self.space.metric.geodesic(
            end_point, end_point=initial_point
        )

        res = geod_func(time)
        res_ = geod_func_reverse(1.0 - time)

        self.assertAllClose(res, res_, atol=atol)

    def test_geodesic_bvp_belongs(self, initial_point, end_point, time, atol=gs.atol):
        """Check geodesic belongs to manifold.

        This is for geodesics defined by the boundary value problem (bvp).

        Parameters
        ----------
        atol : float
            Absolute tolerance.
        """
        geod_func = self.space.metric.geodesic(initial_point, end_point=end_point)
        points = geod_func(time)

        assert_belongs(self.space, points, atol=atol)
