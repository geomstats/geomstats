"""
Unit tests for parameterized manifolds.
"""

import unittest

import geomstats.backend as gs

from geomstats.hypersphere import Hypersphere
from geomstats.discretized_curves_space import DiscretizedCurvesSpace


S2 = Hypersphere(dimension=2)
R3 = S2.embedding_manifold

INITIAL_POINT = [0, 0, 1]
INITIAL_TANGENT_VEC_A = [1, 0, 0]
INITIAL_TANGENT_VEC_B = [0, 1, 0]
INITIAL_TANGENT_VEC_C = [-1, 0, 0]

CURVE_A = S2.metric.geodesic(initial_point=INITIAL_POINT,
                             initial_tangent_vec=INITIAL_TANGENT_VEC_A)
CURVE_B = S2.metric.geodesic(initial_point=INITIAL_POINT,
                             initial_tangent_vec=INITIAL_TANGENT_VEC_B)
CURVE_C = S2.metric.geodesic(initial_point=INITIAL_POINT,
                             initial_tangent_vec=INITIAL_TANGENT_VEC_C)

N_SAMPLING_POINTS = 10
SAMPLING_TIMES = gs.linspace(0, 1, N_SAMPLING_POINTS)
POINT_A = CURVE_A(SAMPLING_TIMES)
POINT_B = CURVE_B(SAMPLING_TIMES)
POINT_C = CURVE_C(SAMPLING_TIMES)

N_POINTS = 5
TIMES = gs.linspace(0, 1, N_POINTS)
ATOL = 1e-8


class TestDiscretizedCurvesSpaceMethods(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        gs.random.seed(1234)
        self.space_curves_R3 = DiscretizedCurvesSpace(embedding_manifold=R3)
        self.space_curves_S2 = DiscretizedCurvesSpace(embedding_manifold=S2)
        self.point_a = POINT_A
        self.point_b = POINT_B
        self.point_c = POINT_C
        points_ab = self.space_curves_S2.L2_metric.geodesic(
                self.point_a, self.point_b)
        points_bc = self.space_curves_S2.L2_metric.geodesic(
                self.point_b, self.point_c)
        self.points_ab = points_ab(TIMES)
        self.points_bc = points_bc(TIMES)
        self.tangent_vec = self.space_curves_S2.L2_metric.log(
                point=self.point_b, base_point=self.point_a)
        self.tangent_vecs = self.space_curves_S2.L2_metric.log(
                point=self.points_bc, base_point=self.points_ab)

    def test_belongs(self):
        result = self.space_curves_S2.belongs(self.point_a)
        self.assertTrue(gs.all(result))

    def test_l2_metric_log_and_squared_norm_and_dist(self):
        """
        Test that squared norm of logarithm is squared dist.
        """
        log_ab = self.tangent_vec
        result = self.space_curves_S2.L2_metric.squared_norm(
                vector=log_ab, base_point=self.point_a)
        expected = self.space_curves_S2.L2_metric.dist(
                self.point_a, self.point_b) ** 2

        gs.testing.assert_allclose(result, expected)

    def test_l2_metric_log_and_exp(self):
        """
        Test that exp and log are inverse maps.
        """
        result = self.space_curves_S2.L2_metric.exp(
                tangent_vec=self.tangent_vec,
                base_point=self.point_a)
        expected = self.point_b

        gs.testing.assert_allclose(result, expected, atol=ATOL)

    def test_l2_metric_inner_product_vectorization(self):
        """
        Test the vectorization inner_product.
        """
        result = self.space_curves_S2.L2_metric.inner_product(
                self.tangent_vecs, self.tangent_vecs, self.points_ab)
        expected = gs.zeros(N_POINTS)
        for k in range(N_POINTS):
            expected[k] = self.space_curves_S2.L2_metric.inner_product(
                    self.tangent_vecs[k, :],
                    self.tangent_vecs[k, :],
                    self.points_ab[k, :])

        gs.testing.assert_allclose(result, expected)

    def test_l2_metric_dist_vectorization(self):
        """
        Test the vectorization of dist.
        """
        result = self.space_curves_S2.L2_metric.dist(
                self.points_ab, self.points_bc)
        expected = gs.zeros(N_POINTS)
        for k in range(N_POINTS):
            expected[k] = self.space_curves_S2.L2_metric.dist(
                    self.points_ab[k, :], self.points_bc[k, :])

        gs.testing.assert_allclose(result, expected)

    def test_l2_metric_exp_vectorization(self):
        """
        Test the vectorization of exp.
        """
        result = self.space_curves_S2.L2_metric.exp(
                tangent_vec=self.tangent_vecs,
                base_point=self.points_ab)
        expected = gs.zeros(self.points_ab.shape)
        for k in range(N_POINTS):
            expected[k, :] = self.space_curves_S2.L2_metric.exp(
                    tangent_vec=self.tangent_vecs[k, :],
                    base_point=self.points_ab[k, :])

        gs.testing.assert_allclose(result, expected)

    def test_l2_metric_log_vectorization(self):
        """
        Test the vectorization of log.
        """
        result = self.tangent_vecs
        expected = gs.zeros(self.points_ab.shape)
        for k in range(N_POINTS):
            expected[k, :] = self.space_curves_S2.L2_metric.log(
                    point=self.points_bc[k, :],
                    base_point=self.points_ab[k, :])

        gs.testing.assert_allclose(result, expected)

    def test_l2_metric_geodesic(self):
        """
        Test the geodesic method of L2Metric.
        """
        result = self.points_ab
        expected = gs.zeros(self.points_ab.shape)
        for k in range(N_SAMPLING_POINTS):
            geod = self.space_curves_S2.L2_metric.embedding_metric.\
                geodesic(initial_point=self.point_a[k, :],
                         end_point=self.point_b[k, :])
            expected[:, k, :] = geod(TIMES)

        gs.testing.assert_allclose(result, expected)

        geod = self.space_curves_S2.L2_metric.geodesic(
                initial_point=self.points_ab,
                end_point=self.points_bc)

    def test_srv_metric_pointwise_inner_product(self):
        result = self.space_curves_R3.SRV_metric.pointwise_inner_product(
                tangent_vec_a=self.tangent_vecs,
                tangent_vec_b=self.tangent_vecs,
                base_point=self.points_ab)
        expected_shape = [N_POINTS, N_SAMPLING_POINTS]
        gs.testing.assert_allclose(result.shape, expected_shape)

    def test_square_root_velocity_and_inverse(self):
        """
        Test of square_root_velocity and its inverse.
        N.B: Here points_ab are seen as curves in R3 and not S2.
        """
        curves = self.points_ab
        srv_curves = self.space_curves_R3.SRV_metric.square_root_velocity(
                curves)
        starting_points = curves[:, [0], :]
        result = self.space_curves_R3.SRV_metric.square_root_velocity_inverse(
                srv_curves, starting_points)
        expected = curves

        gs.testing.assert_allclose(result, expected)

    def test_srv_metric_exp_and_log(self):
        """
        Test that exp and log are inverse maps and vectorized.
        N.B: Here points_ab and points_bc are seen as curves in R3 and not S2.
        """
        log = self.space_curves_R3.SRV_metric.log(point=self.points_bc,
                                                  base_point=self.points_ab)
        result = self.space_curves_R3.SRV_metric.exp(tangent_vec=log,
                                                     base_point=self.points_ab)
        expected = self.points_bc

        gs.testing.assert_allclose(result.squeeze(), expected, atol=ATOL)

    def test_srv_metric_geodesic(self):
        """
        Test that the geodesic between two curves in a Euclidean space
        for the srv metric is the L2 geodesic betweeen the curves srvs.
        N.B: Here point_a and point_b are seen as curves in R3 and not S2.
        """
        geod = self.space_curves_R3.SRV_metric.geodesic(
                initial_point=self.point_a,
                end_point=self.point_b)
        result = geod(TIMES)

        srv_a = self.space_curves_R3.SRV_metric.square_root_velocity(
                self.point_a)
        srv_b = self.space_curves_R3.SRV_metric.square_root_velocity(
                self.point_b)
        geod_srv = self.space_curves_R3.L2_metric.geodesic(
                initial_point=srv_a,
                end_point=srv_b)
        geod_srv = geod_srv(TIMES)

        origin = self.space_curves_R3.SRV_metric.embedding_metric.geodesic(
                initial_point=self.point_a[0, :],
                end_point=self.point_b[0, :])
        origin = origin(TIMES)

        expected = self.space_curves_R3.SRV_metric.\
            square_root_velocity_inverse(geod_srv, origin)

        gs.testing.assert_allclose(result, expected, atol=ATOL)

    def test_srv_metric_dist_and_geod(self):
        """
        Test that the length of the geodesic gives the distance.
        N.B: Here point_a and point_b are seen as curves in R3 and not S2.
        """
        geod = self.space_curves_R3.SRV_metric.geodesic(
                initial_point=self.point_a,
                end_point=self.point_b)
        geod = geod(TIMES)

        srv = self.space_curves_R3.SRV_metric.square_root_velocity(geod)

        srv_derivative = N_POINTS * (srv[1:, :] - srv[:-1, :])
        result = self.space_curves_R3.L2_metric.norm(srv_derivative,
                                                     geod[:-1, :-1, :])
        result = gs.sum(result, 0) / N_POINTS
        expected = self.space_curves_R3.SRV_metric.dist(self.point_a,
                                                        self.point_b)

        gs.testing.assert_allclose(result, expected)


if __name__ == '__main__':
        unittest.main()
