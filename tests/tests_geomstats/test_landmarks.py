"""Unit tests for landmarks space."""


import geomstats.backend as gs
import geomstats.tests
from tests.conftest import Parametrizer
from tests.data.landmarks_data import LandmarksTestData, TestDataL2Metric
from tests.geometry_test_cases import ManifoldTestCase, RiemannianMetricTestCase


class TestLandmarks(ManifoldTestCase, metaclass=Parametrizer):
    skip_test_random_point_belongs = True
    skip_test_random_tangent_vec_is_tangent = True

    testing_data = LandmarksTestData()
    space = testing_data.space


class TestL2Metric(RiemannianMetricTestCase, metaclass=Parametrizer):
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_exp_geodesic_ivp = True
    skip_test_exp_belongs = True
    skip_test_exp_shape = True
    skip_test_log_shape = True

    testing_data = TestDataL2Metric()
    Metric = Connection = testing_data.Metric

    @geomstats.tests.np_autograd_and_tf_only
    def test_l2_metric_inner_product_vectorization(
        self,
        l2_metric_s2,
        times,
        n_landmark_sets,
        landmarks_a,
        landmarks_b,
        landmarks_c,
    ):
        """Test the vectorization inner_product."""
        landmarks_ab = l2_metric_s2.geodesic(landmarks_a, landmarks_b)
        landmarks_bc = l2_metric_s2.geodesic(landmarks_b, landmarks_c)
        landmarks_ab = landmarks_ab(times)
        landmarks_bc = landmarks_bc(times)

        tangent_vecs = l2_metric_s2.log(point=landmarks_bc, base_point=landmarks_ab)

        result = l2_metric_s2.inner_product(tangent_vecs, tangent_vecs, landmarks_ab)

        self.assertAllClose(gs.shape(result), (n_landmark_sets,))

    @geomstats.tests.np_autograd_and_tf_only
    def test_l2_metric_exp_vectorization(
        self, l2_metric_s2, times, landmarks_a, landmarks_b, landmarks_c
    ):
        """Test the vectorization of exp."""
        landmarks_ab = l2_metric_s2.geodesic(landmarks_a, landmarks_b)
        landmarks_bc = l2_metric_s2.geodesic(landmarks_b, landmarks_c)
        landmarks_ab = landmarks_ab(times)
        landmarks_bc = landmarks_bc(times)

        tangent_vecs = l2_metric_s2.log(point=landmarks_bc, base_point=landmarks_ab)

        result = l2_metric_s2.exp(tangent_vec=tangent_vecs, base_point=landmarks_ab)
        self.assertAllClose(gs.shape(result), gs.shape(landmarks_ab))

    @geomstats.tests.np_autograd_and_tf_only
    def test_l2_metric_log_vectorization(
        self, l2_metric_s2, times, landmarks_a, landmarks_b, landmarks_c
    ):
        """Test the vectorization of log."""
        landmarks_ab = l2_metric_s2.geodesic(landmarks_a, landmarks_b)
        landmarks_bc = l2_metric_s2.geodesic(landmarks_b, landmarks_c)
        landmarks_ab = landmarks_ab(times)
        landmarks_bc = landmarks_bc(times)

        tangent_vecs = l2_metric_s2.log(point=landmarks_bc, base_point=landmarks_ab)

        result = tangent_vecs
        self.assertAllClose(gs.shape(result), gs.shape(landmarks_ab))

    @geomstats.tests.np_autograd_and_tf_only
    def test_l2_metric_geodesic(
        self, l2_metric_s2, times, n_sampling_points, landmarks_a, landmarks_b
    ):
        """Test the geodesic method of L2Metric."""
        landmarks_ab = l2_metric_s2.geodesic(landmarks_a, landmarks_b)
        landmarks_ab = landmarks_ab(times)

        result = landmarks_ab
        expected = []
        for k in range(n_sampling_points):
            geod = l2_metric_s2.ambient_metric.geodesic(
                initial_point=landmarks_a[k, :], end_point=landmarks_b[k, :]
            )
            expected.append(geod(times))
        expected = gs.stack(expected, axis=1)

        self.assertAllClose(result, expected)
