"""
Unit tests for parameterized manifolds.
"""

import geomstats.tests

import geomstats.backend as gs

from geomstats.discretized_curves_space import DiscretizedCurvesSpace
from geomstats.hypersphere import Hypersphere


class TestDiscretizedCurvesSpaceMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        S2 = Hypersphere(dimension=2)
        R3 = S2.embedding_manifold

        INITIAL_POINT = [0., 0., 1.]
        INITIAL_TANGENT_VEC_A = [1., 0., 0.]
        INITIAL_TANGENT_VEC_B = [0., 1., 0.]
        INITIAL_TANGENT_VEC_C = [-1., 0., 0.]

        CURVE_A = S2.metric.geodesic(initial_point=INITIAL_POINT,
                                     initial_tangent_vec=INITIAL_TANGENT_VEC_A)
        CURVE_B = S2.metric.geodesic(initial_point=INITIAL_POINT,
                                     initial_tangent_vec=INITIAL_TANGENT_VEC_B)
        CURVE_C = S2.metric.geodesic(initial_point=INITIAL_POINT,
                                     initial_tangent_vec=INITIAL_TANGENT_VEC_C)

        self.N_SAMPLING_POINTS = 10
        SAMPLING_TIMES = gs.linspace(0., 1., self.N_SAMPLING_POINTS)
        DISCRETIZED_CURVE_A = CURVE_A(SAMPLING_TIMES)
        DISCRETIZED_CURVE_B = CURVE_B(SAMPLING_TIMES)
        DISCRETIZED_CURVE_C = CURVE_C(SAMPLING_TIMES)

        self.N_DISCRETIZED_CURVES = 5
        self.TIMES = gs.linspace(0., 1., self.N_DISCRETIZED_CURVES)
        self.ATOL = 1e-8
        gs.random.seed(1234)
        self.space_curves_in_euclidean_3d = DiscretizedCurvesSpace(
                embedding_manifold=R3)
        self.space_curves_in_sphere_2d = DiscretizedCurvesSpace(
                embedding_manifold=S2)
        self.l2_metric_s2 = self.space_curves_in_sphere_2d.l2_metric
        self.l2_metric_r3 = self.space_curves_in_euclidean_3d.l2_metric
        self.srv_metric_r3 = self.space_curves_in_euclidean_3d.\
            square_root_velocity_metric
        self.curve_a = DISCRETIZED_CURVE_A
        self.curve_b = DISCRETIZED_CURVE_B
        self.curve_c = DISCRETIZED_CURVE_C

    @geomstats.tests.np_only
    def test_belongs(self):
        result = self.space_curves_in_sphere_2d.belongs(self.curve_a)
        self.assertTrue(gs.all(result))

    @geomstats.tests.np_only
    def test_l2_metric_log_and_squared_norm_and_dist(self):
        """
        Test that squared norm of logarithm is squared dist.
        """
        tangent_vec = self.l2_metric_s2.log(
                curve=self.curve_b, base_curve=self.curve_a)
        log_ab = tangent_vec
        result = self.l2_metric_s2.squared_norm(
                vector=log_ab, base_point=self.curve_a)
        expected = self.l2_metric_s2.dist(self.curve_a, self.curve_b) ** 2

        gs.testing.assert_allclose(result, expected)

    @geomstats.tests.np_only
    def test_l2_metric_log_and_exp(self):
        """
        Test that exp and log are inverse maps.
        """
        tangent_vec = self.l2_metric_s2.log(
                curve=self.curve_b, base_curve=self.curve_a)
        result = self.l2_metric_s2.exp(tangent_vec=tangent_vec,
                                       base_curve=self.curve_a)
        expected = self.curve_b

        gs.testing.assert_allclose(result, expected, atol=self.ATOL)

    @geomstats.tests.np_only
    def test_l2_metric_inner_product_vectorization(self):
        """
        Test the vectorization inner_product.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.TIMES)
        curves_bc = curves_bc(self.TIMES)

        tangent_vecs = self.l2_metric_s2.log(
                curve=curves_bc, base_curve=curves_ab)

        result = self.l2_metric_s2.inner_product(
                tangent_vecs, tangent_vecs, curves_ab)
        expected = gs.zeros(self.N_DISCRETIZED_CURVES)
        for k in range(self.N_DISCRETIZED_CURVES):
            expected[k] = self.l2_metric_s2.inner_product(
                    tangent_vecs[k, :],
                    tangent_vecs[k, :],
                    curves_ab[k, :])

        gs.testing.assert_allclose(result, expected)

    @geomstats.tests.np_only
    def test_l2_metric_dist_vectorization(self):
        """
        Test the vectorization of dist.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.TIMES)
        curves_bc = curves_bc(self.TIMES)

        result = self.l2_metric_s2.dist(
                curves_ab, curves_bc)
        expected = gs.zeros(self.N_DISCRETIZED_CURVES)
        for k in range(self.N_DISCRETIZED_CURVES):
            expected[k] = self.l2_metric_s2.dist(
                    curves_ab[k, :], curves_bc[k, :])

        gs.testing.assert_allclose(result, expected)

    @geomstats.tests.np_only
    def test_l2_metric_exp_vectorization(self):
        """
        Test the vectorization of exp.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.TIMES)
        curves_bc = curves_bc(self.TIMES)

        tangent_vecs = self.l2_metric_s2.log(
                curve=curves_bc, base_curve=curves_ab)

        result = self.l2_metric_s2.exp(
                tangent_vec=tangent_vecs,
                base_curve=curves_ab)
        expected = gs.zeros(curves_ab.shape)
        for k in range(self.N_DISCRETIZED_CURVES):
            expected[k, :] = self.l2_metric_s2.exp(
                    tangent_vec=tangent_vecs[k, :],
                    base_curve=curves_ab[k, :])

        gs.testing.assert_allclose(result, expected)

    @geomstats.tests.np_only
    def test_l2_metric_log_vectorization(self):
        """
        Test the vectorization of log.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.TIMES)
        curves_bc = curves_bc(self.TIMES)

        tangent_vecs = self.l2_metric_s2.log(
                curve=curves_bc, base_curve=curves_ab)

        result = tangent_vecs
        expected = gs.zeros(curves_ab.shape)
        for k in range(self.N_DISCRETIZED_CURVES):
            expected[k, :] = self.l2_metric_s2.log(
                    curve=curves_bc[k, :],
                    base_curve=curves_ab[k, :])

        gs.testing.assert_allclose(result, expected)

    @geomstats.tests.np_only
    def test_l2_metric_geodesic(self):
        """
        Test the geodesic method of L2Metric.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.TIMES)
        curves_bc = curves_bc(self.TIMES)

        result = curves_ab
        expected = gs.zeros(curves_ab.shape)
        for k in range(self.N_SAMPLING_POINTS):
            geod = self.l2_metric_s2.embedding_metric.geodesic(
                    initial_point=self.curve_a[k, :],
                    end_point=self.curve_b[k, :])
            expected[:, k, :] = geod(self.TIMES)

        gs.testing.assert_allclose(result, expected)

        geod = self.l2_metric_s2.geodesic(
                initial_curve=curves_ab,
                end_curve=curves_bc)

    @geomstats.tests.np_only
    def test_srv_metric_pointwise_inner_product(self):
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.TIMES)
        curves_bc = curves_bc(self.TIMES)

        tangent_vecs = self.l2_metric_s2.log(
                curve=curves_bc, base_curve=curves_ab)

        result = self.srv_metric_r3.pointwise_inner_product(
                tangent_vec_a=tangent_vecs,
                tangent_vec_b=tangent_vecs,
                base_curve=curves_ab)
        expected_shape = [self.N_DISCRETIZED_CURVES, self.N_SAMPLING_POINTS]
        gs.testing.assert_allclose(result.shape, expected_shape)

    @geomstats.tests.np_only
    def test_square_root_velocity_and_inverse(self):
        """
        Test of square_root_velocity and its inverse.
        N.B: Here curves_ab are seen as curves in R3 and not S2.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_ab = curves_ab(self.TIMES)

        curves = curves_ab
        srv_curves = self.srv_metric_r3.square_root_velocity(curves)
        starting_points = curves[:, [0], :]
        result = self.srv_metric_r3.square_root_velocity_inverse(
                srv_curves, starting_points)
        expected = curves

        gs.testing.assert_allclose(result, expected)

    @geomstats.tests.np_only
    def test_srv_metric_exp_and_log(self):
        """
        Test that exp and log are inverse maps and vectorized.
        N.B: Here curves_ab and curves_bc are seen as curves in R3 and not S2.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.TIMES)
        curves_bc = curves_bc(self.TIMES)

        log = self.srv_metric_r3.log(curve=curves_bc,
                                     base_curve=curves_ab)
        result = self.srv_metric_r3.exp(tangent_vec=log,
                                        base_curve=curves_ab)
        expected = curves_bc

        gs.testing.assert_allclose(result.squeeze(), expected, atol=self.ATOL)

    @geomstats.tests.np_only
    def test_srv_metric_geodesic(self):
        """
        Test that the geodesic between two curves in a Euclidean space
        for the srv metric is the L2 geodesic betweeen the curves srvs.
        N.B: Here curve_a and curve_b are seen as curves in R3 and not S2.
        """
        geod = self.srv_metric_r3.geodesic(
                initial_curve=self.curve_a,
                end_curve=self.curve_b)
        result = geod(self.TIMES)

        srv_a = self.srv_metric_r3.square_root_velocity(self.curve_a)
        srv_b = self.srv_metric_r3.square_root_velocity(self.curve_b)
        geod_srv = self.l2_metric_r3.geodesic(initial_curve=srv_a,
                                              end_curve=srv_b)
        geod_srv = geod_srv(self.TIMES)

        starting_points = self.srv_metric_r3.embedding_metric.geodesic(
                initial_point=self.curve_a[0, :],
                end_point=self.curve_b[0, :])
        starting_points = starting_points(self.TIMES)

        expected = self.srv_metric_r3.square_root_velocity_inverse(
                geod_srv, starting_points)

        gs.testing.assert_allclose(result, expected, atol=self.ATOL)

    @geomstats.tests.np_only
    def test_srv_metric_dist_and_geod(self):
        """
        Test that the length of the geodesic gives the distance.
        N.B: Here curve_a and curve_b are seen as curves in R3 and not S2.
        """
        geod = self.srv_metric_r3.geodesic(initial_curve=self.curve_a,
                                           end_curve=self.curve_b)
        geod = geod(self.TIMES)

        srv = self.srv_metric_r3.square_root_velocity(geod)

        srv_derivative = self.N_DISCRETIZED_CURVES * (srv[1:, :] - srv[:-1, :])
        result = self.l2_metric_r3.norm(srv_derivative, geod[:-1, :-1, :])
        result = gs.sum(result, 0) / self.N_DISCRETIZED_CURVES
        expected = self.srv_metric_r3.dist(self.curve_a, self.curve_b)

        gs.testing.assert_allclose(result, expected)


if __name__ == '__main__':
        geomstats.tests.main()
