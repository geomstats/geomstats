"""Unit tests for the pull-back diffeo metrics."""

from geomstats.tests import autograd_backend, pytorch_backend
from tests.conftest import Parametrizer
from tests.data.pullback_diffeo_metric_data import (
    CircleAsSO2Metric,
    CircleAsSO2MetricTestData,
    CircleAsSO2PullbackDiffeoMetricTestData,
)
from tests.geometry_test_cases import PullbackDiffeoMetricTestCase
from tests.tests_geomstats.test_hypersphere import HypersphereMetricTestCase


class TestHypersphereBisMetric(HypersphereMetricTestCase, metaclass=Parametrizer):
    metric = connection = CircleAsSO2Metric

    skip_test_exp_geodesic_ivp = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_christoffels_shape = True
    skip_test_sectional_curvature = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_all = not (autograd_backend() or pytorch_backend())
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = CircleAsSO2MetricTestData()


class TestCircleAsSO2PullbackDiffeoMetric(
    PullbackDiffeoMetricTestCase, metaclass=Parametrizer
):

    metric = CircleAsSO2Metric

    skip_all = not (autograd_backend() or pytorch_backend())
    skip_test_covariant_riemann_tensor_is_skew_symmetric_1 = True
    skip_test_covariant_riemann_tensor_is_skew_symmetric_2 = True
    skip_test_covariant_riemann_tensor_bianchi_identity = True
    skip_test_covariant_riemann_tensor_is_interchange_symmetric = True
    skip_test_riemann_tensor_shape = True
    skip_test_scalar_curvature_shape = True
    skip_test_ricci_tensor_shape = True
    skip_test_sectional_curvature_shape = True

    testing_data = CircleAsSO2PullbackDiffeoMetricTestData()
