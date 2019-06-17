"""
Unit tests for parameterized manifolds.
"""

import geomstats.backend as gs
import geomstats.tests
import tests.helper as helper

from geomstats.discretized_curves_space import DiscretizedCurvesSpace
from geomstats.hypersphere import Hypersphere


class TestDiscretizedCurvesSpaceMethods(geomstats.tests.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        s2 = Hypersphere(dimension=2)
        r3 = s2.embedding_manifold

        initial_point = [0., 0., 1.]
        initial_tangent_vec_a = [1., 0., 0.]
        initial_tangent_vec_b = [0., 1., 0.]
        initial_tangent_vec_c = [-1., 0., 0.]

        curve_a = s2.metric.geodesic(initial_point=initial_point,
                                     initial_tangent_vec=initial_tangent_vec_a)
        curve_b = s2.metric.geodesic(initial_point=initial_point,
                                     initial_tangent_vec=initial_tangent_vec_b)
        curve_c = s2.metric.geodesic(initial_point=initial_point,
                                     initial_tangent_vec=initial_tangent_vec_c)

        self.n_sampling_points = 10
        sampling_times = gs.linspace(0., 1., self.n_sampling_points)
        discretized_curve_a = curve_a(sampling_times)
        discretized_curve_b = curve_b(sampling_times)
        discretized_curve_c = curve_c(sampling_times)

        self.n_discretized_curves = 5
        self.times = gs.linspace(0., 1., self.n_discretized_curves)
        self.atol = 1e-6
        gs.random.seed(1234)
        self.space_curves_in_euclidean_3d = DiscretizedCurvesSpace(
                embedding_manifold=r3)
        self.space_curves_in_sphere_2d = DiscretizedCurvesSpace(
                embedding_manifold=s2)
        self.l2_metric_s2 = self.space_curves_in_sphere_2d.l2_metric
        self.l2_metric_r3 = self.space_curves_in_euclidean_3d.l2_metric
        self.srv_metric_r3 = self.space_curves_in_euclidean_3d.\
            square_root_velocity_metric
        self.curve_a = discretized_curve_a
        self.curve_b = discretized_curve_b
        self.curve_c = discretized_curve_c

    def test_belongs(self):
        result = self.space_curves_in_sphere_2d.belongs(self.curve_a)
        expected = gs.array([[True]])

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
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
        expected = helper.to_scalar(expected)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_and_tf_only
    def test_l2_metric_log_and_exp(self):
        """
        Test that exp and log are inverse maps.
        """
        tangent_vec = self.l2_metric_s2.log(
                curve=self.curve_b, base_curve=self.curve_a)
        result = self.l2_metric_s2.exp(tangent_vec=tangent_vec,
                                       base_curve=self.curve_a)
        expected = self.curve_b

        self.assertAllClose(result, expected, atol=self.atol)

    @geomstats.tests.np_and_tf_only
    def test_l2_metric_inner_product_vectorization(self):
        """
        Test the vectorization inner_product.
        """
        n_samples = self.n_discretized_curves
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        tangent_vecs = self.l2_metric_s2.log(
                curve=curves_bc, base_curve=curves_ab)

        result = self.l2_metric_s2.inner_product(
                tangent_vecs, tangent_vecs, curves_ab)

        self.assertAllClose(gs.shape(result), (n_samples, 1))

    @geomstats.tests.np_and_tf_only
    def test_l2_metric_dist_vectorization(self):
        """
        Test the vectorization of dist.
        """
        n_samples = self.n_discretized_curves
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        result = self.l2_metric_s2.dist(
                curves_ab, curves_bc)
        self.assertAllClose(gs.shape(result), (n_samples, 1))

    @geomstats.tests.np_and_tf_only
    def test_l2_metric_exp_vectorization(self):
        """
        Test the vectorization of exp.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        tangent_vecs = self.l2_metric_s2.log(
                curve=curves_bc, base_curve=curves_ab)

        result = self.l2_metric_s2.exp(
                tangent_vec=tangent_vecs,
                base_curve=curves_ab)
        self.assertAllClose(gs.shape(result), gs.shape(curves_ab))

    @geomstats.tests.np_and_tf_only
    def test_l2_metric_log_vectorization(self):
        """
        Test the vectorization of log.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        tangent_vecs = self.l2_metric_s2.log(
                curve=curves_bc, base_curve=curves_ab)

        result = tangent_vecs
        self.assertAllClose(gs.shape(result), gs.shape(curves_ab))

    @geomstats.tests.np_only
    def test_l2_metric_geodesic(self):
        """
        Test the geodesic method of L2Metric.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        result = curves_ab
        expected = gs.zeros(curves_ab.shape)
        for k in range(self.n_sampling_points):
            geod = self.l2_metric_s2.embedding_metric.geodesic(
                    initial_point=self.curve_a[k, :],
                    end_point=self.curve_b[k, :])
            expected[:, k, :] = geod(self.times)

        gs.testing.assert_allclose(result, expected)

        geod = self.l2_metric_s2.geodesic(
                initial_curve=curves_ab,
                end_curve=curves_bc)

    @geomstats.tests.np_and_tf_only
    def test_srv_metric_pointwise_inner_product(self):
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        tangent_vecs = self.l2_metric_s2.log(
                curve=curves_bc, base_curve=curves_ab)

        result = self.srv_metric_r3.pointwise_inner_product(
                tangent_vec_a=tangent_vecs,
                tangent_vec_b=tangent_vecs,
                base_curve=curves_ab)
        expected_shape = (self.n_discretized_curves, self.n_sampling_points)
        self.assertAllClose(gs.shape(result), expected_shape)

    @geomstats.tests.np_only
    def test_square_root_velocity_and_inverse(self):
        """
        Test of square_root_velocity and its inverse.
        N.B: Here curves_ab are seen as curves in R3 and not S2.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_ab = curves_ab(self.times)

        curves = curves_ab
        srv_curves = self.srv_metric_r3.square_root_velocity(curves)
        starting_points = curves[:, 0, :]
        result = self.srv_metric_r3.square_root_velocity_inverse(
                srv_curves, starting_points)
        expected = curves

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_srv_metric_exp_and_log(self):
        """
        Test that exp and log are inverse maps and vectorized.
        N.B: Here curves_ab and curves_bc are seen as curves in R3 and not S2.
        """
        curves_ab = self.l2_metric_s2.geodesic(self.curve_a, self.curve_b)
        curves_bc = self.l2_metric_s2.geodesic(self.curve_b, self.curve_c)
        curves_ab = curves_ab(self.times)
        curves_bc = curves_bc(self.times)

        log = self.srv_metric_r3.log(curve=curves_bc,
                                     base_curve=curves_ab)
        result = self.srv_metric_r3.exp(tangent_vec=log,
                                        base_curve=curves_ab)
        expected = curves_bc

        self.assertAllClose(gs.squeeze(result), expected)

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
        result = geod(self.times)

        srv_a = self.srv_metric_r3.square_root_velocity(self.curve_a)
        srv_b = self.srv_metric_r3.square_root_velocity(self.curve_b)
        geod_srv = self.l2_metric_r3.geodesic(initial_curve=srv_a,
                                              end_curve=srv_b)
        geod_srv = geod_srv(self.times)

        starting_points = self.srv_metric_r3.embedding_metric.geodesic(
                initial_point=self.curve_a[0, :],
                end_point=self.curve_b[0, :])
        starting_points = starting_points(self.times)

        expected = self.srv_metric_r3.square_root_velocity_inverse(
                geod_srv, starting_points)

        self.assertAllClose(result, expected)

    @geomstats.tests.np_only
    def test_srv_metric_dist_and_geod(self):
        """
        Test that the length of the geodesic gives the distance.
        N.B: Here curve_a and curve_b are seen as curves in R3 and not S2.
        """
        geod = self.srv_metric_r3.geodesic(initial_curve=self.curve_a,
                                           end_curve=self.curve_b)
        geod = geod(self.times)

        srv = self.srv_metric_r3.square_root_velocity(geod)

        srv_derivative = self.n_discretized_curves * (srv[1:, :] - srv[:-1, :])
        result = self.l2_metric_r3.norm(srv_derivative, geod[:-1, :-1, :])
        result = gs.sum(result, 0) / self.n_discretized_curves
        expected = self.srv_metric_r3.dist(self.curve_a, self.curve_b)

        self.assertAllClose(result, expected)


if __name__ == '__main__':
        geomstats.tests.main()
