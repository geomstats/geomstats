"""Unit tests for the pull-back diffeo metrics."""

import random

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from tests.conftest import Parametrizer
from tests.data.hypersphere_data import HypersphereMetricTestData
from tests.data.pullback_diffeo_metric_data import PullbackDiffeoCircleMetricTestData
from tests.geometry_test_cases import PullbackDiffeoMetricTestCase
from tests.tests_geomstats.test_hypersphere import TestHypersphereMetric

RTOL = 1e-5
ATOL = 1e-6


class CircleAsSpecialOrthogonalMetric(PullbackDiffeoMetric):
    def __init__(self, **kwargs):
        super(CircleAsSpecialOrthogonalMetric, self).__init__(dim=1, shape=(2,))

    def create_embedding_metric(self):
        return SpecialOrthogonal(n=2, point_type="matrix").bi_invariant_metric

    def diffeomorphism(self, base_point):
        second_column = gs.stack([-base_point[..., 1], base_point[..., 0]], axis=-1)
        return gs.stack([base_point, second_column], axis=-1)

    def inverse_diffeomorphism(self, image_point):
        return image_point[..., 0]


class HypersphereBis(Hypersphere):
    def __init__(self, dim, default_coords_type="extrinsic"):
        assert dim == 1
        assert default_coords_type == "extrinsic"
        super(HypersphereBis, self).__init__(dim, default_coords_type)
        self._metric = CircleAsSpecialOrthogonalMetric()


pb_circle_metric = CircleAsSpecialOrthogonalMetric()
natural_circle_metric = Hypersphere(1).metric


# First we run all test of the hypersphere with the new way


class HypersphereBisMetricTestData(HypersphereMetricTestData):
    dim_list = [1] * 4
    metric_args_list = [(n,) for n in dim_list]
    shape_list = [(dim + 1,) for dim in dim_list]
    space_list = [HypersphereBis(n) for n in dim_list]
    n_points_list = random.sample(range(1, 5), 4)
    n_tangent_vecs_list = random.sample(range(1, 5), 4)
    n_points_a_list = random.sample(range(1, 5), 4)
    n_points_b_list = [1]
    alpha_list = [1] * 4
    n_rungs_list = [1] * 4
    scheme_list = ["pole"] * 4


@geomstats.tests.np_autograd_and_torch_only
class TestHypersphereBisMetric(TestHypersphereMetric, metaclass=Parametrizer):
    metric = connection = CircleAsSpecialOrthogonalMetric
    skip_test_exp_geodesic_ivp = True
    skip_test_dist_point_to_itself_is_zero = True

    testing_data = HypersphereBisMetricTestData()


# Then we run PullbackDiffeo test


@geomstats.tests.np_autograd_and_torch_only
class TestPullbackDiffeoCircle(PullbackDiffeoMetricTestCase, metaclass=Parametrizer):

    metric = CircleAsSpecialOrthogonalMetric

    testing_data = PullbackDiffeoCircleMetricTestData()

    def test_diffeomorphism_is_reciprocal(self, point):
        super().test_diffeomorphism_is_reciprocal([], point, RTOL, ATOL)

    def test_tangent_diffeomorphism_is_reciprocal(self, point, tangent_vector):
        super().test_tangent_diffeomorphism_is_reciprocal(
            [], point, tangent_vector, RTOL, ATOL
        )

    def test_matrix_innerproduct_and_embedded_innerproduct_coincide(self):
        pass
