"""Unit tests for the pull-back diffeo metrics."""

from geomstats.tests import autograd_backend, pytorch_backend
from tests.conftest import Parametrizer
from tests.data.pullback_diffeo_metric_data import (
    CircleAsSO2MetricTestData,
    CircleAsSO2PullbackDiffeoMetricTestData,
)
from tests.geometry_test_cases import PullbackDiffeoMetricTestCase
from tests.tests_geomstats.test_hypersphere import HypersphereMetricTestCase


class TestHypersphereBisMetric(HypersphereMetricTestCase, metaclass=Parametrizer):

    skip_test_exp_geodesic_ivp = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_christoffels_shape = True
    skip_test_sectional_curvature = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_parallel_transport_ivp_is_isometry = True
    skip_all = not (autograd_backend() or pytorch_backend())

    testing_data = CircleAsSO2MetricTestData()


class TestCircleAsSO2PullbackDiffeoMetric(
    PullbackDiffeoMetricTestCase, metaclass=Parametrizer
):
    skip_all = not (autograd_backend() or pytorch_backend())

    testing_data = CircleAsSO2PullbackDiffeoMetricTestData()
