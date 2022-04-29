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
from tests.tests_geomstats.test_hypersphere import AbstractHypersphereMetric


class CircleAsSpecialOrthogonalMetric(PullbackDiffeoMetric):
    def __init__(self, *args, **kwargs):
        super(CircleAsSpecialOrthogonalMetric, self).__init__(dim=1, shape=(2,))

    def create_embedding_metric(self):
        return SpecialOrthogonal(n=2, point_type="matrix").bi_invariant_metric

    def diffeomorphism(self, base_point):
        second_column = gs.stack([-base_point[..., 1], base_point[..., 0]], axis=-1)
        return gs.stack([base_point, second_column], axis=-1)

    def inverse_diffeomorphism(self, image_point):
        return image_point[..., 0]


# First we run all test of the hypersphere with the new way


class HypersphereBis(Hypersphere):
    def __init__(self, dim, default_coords_type="extrinsic"):
        assert dim == 1
        assert default_coords_type == "extrinsic"
        super(HypersphereBis, self).__init__(dim, default_coords_type)
        self._metric = CircleAsSpecialOrthogonalMetric()


class HypersphereBisMetricTestData(HypersphereMetricTestData):
    dim_list = [1] * 4
    metric_args_list = [tuple([]) for n in dim_list]
    shape_list = [(dim + 1,) for dim in dim_list]
    space_list = [HypersphereBis(n) for n in dim_list]
    n_points_list = random.sample(range(1, 5), 4)
    n_tangent_vecs_list = random.sample(range(1, 5), 4)
    n_points_a_list = [3] * 4
    n_points_b_list = [1]
    alpha_list = [1] * 4
    n_rungs_list = [1] * 4
    scheme_list = ["pole"] * 4

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                dim=1,
                tangent_vec_a=[1.0, 0.0],
                tangent_vec_b=[2.0, 0.0],
                base_point=[0.0, 1.0],
                expected=2.0,
            )
        ]
        return self.generate_tests(smoke_data)

    def dist_test_data(self):
        # smoke data is currently testing points at orthogonal
        point_a = gs.array([10.0, -2.0])
        point_a = point_a / gs.linalg.norm(point_a)
        point_b = gs.array([2.0, 10])
        point_b = point_b / gs.linalg.norm(point_b)
        smoke_data = [dict(dim=1, point_a=point_a, point_b=point_b, expected=gs.pi / 2)]
        return self.generate_tests(smoke_data)

    def diameter_test_data(self):
        point_a = gs.array([[0.0, 1.0]])
        point_b = gs.array([[1.0, 0.0]])
        point_c = gs.array([[1.0, 0.0]])
        smoke_data = [
            dict(
                dim=1, points=gs.vstack((point_a, point_b, point_c)), expected=gs.pi / 2
            )
        ]
        return self.generate_tests(smoke_data)

    def christoffels_shape_test_data(self):
        point = gs.array([[gs.pi / 2], [gs.pi / 6]])
        smoke_data = [dict(dim=1, point=point, expected=[2, 1, 1, 1])]
        return self.generate_tests(smoke_data)

    def sectional_curvature_test_data(self):
        dim_list = [1]
        n_samples_list = random.sample(range(1, 4), 2)
        random_data = []
        for dim, n_samples in zip(dim_list, n_samples_list):
            sphere = Hypersphere(dim)
            base_point = sphere.random_uniform()
            tangent_vec_a = sphere.to_tangent(
                gs.random.rand(n_samples, sphere.dim + 1), base_point
            )
            tangent_vec_b = sphere.to_tangent(
                gs.random.rand(n_samples, sphere.dim + 1), base_point
            )
            expected = gs.ones(n_samples)  # try shape here
            random_data.append(
                dict(
                    dim=dim,
                    tangent_vec_a=tangent_vec_a,
                    tangent_vec_b=tangent_vec_b,
                    base_point=base_point,
                    expected=expected,
                ),
            )
        return self.generate_tests(random_data)

    def dist_pairwise_test_data(self):
        smoke_data = [
            dict(
                dim=1,
                point=[
                    gs.array([1.0, 0.0]),
                    gs.array([0.0, 1.0]),
                ],
                expected=gs.array([[0.0, gs.pi / 2], [gs.pi / 2, 0.0]]),
                rtol=1e-3,
            )
        ]
        return self.generate_tests(smoke_data)

    def exp_after_log_test_data(self):
        # edge case: two very close points, base_point_2 and point_2,
        # form an angle < epsilon
        base_point = gs.array([1.0, 2.0])
        base_point = base_point / gs.linalg.norm(base_point)
        point = base_point + 1e-4 * gs.array([-1.0, -2.0])
        point = point / gs.linalg.norm(point)
        smoke_data = [
            dict(
                space_args=(1,),
                point=point,
                base_point=base_point,
                rtol=gs.rtol,
                atol=gs.atol,
            )
        ]
        return self._exp_after_log_test_data(
            self.metric_args_list,
            self.space_list,
            self.n_points_list,
            smoke_data,
            atol=1e-3,
        )

    def log_after_exp_test_data(self):
        base_point = gs.array([1.0, 0.0])
        tangent_vec = gs.array([0.0, gs.pi / 6])

        smoke_data = [
            dict(
                space_args=(1,),
                tangent_vec=tangent_vec,
                base_point=base_point,
                rtol=gs.rtol * 100,
                atol=gs.atol * 100,
            )
        ]
        return self._log_after_exp_test_data(
            self.metric_args_list,
            self.space_list,
            self.shape_list,
            self.n_tangent_vecs_list,
            smoke_data,
            rtol=gs.rtol * 100,
            atol=gs.atol * 100,
        )

    def exp_and_dist_and_projection_to_tangent_space_test_data(self):
        unnorm_base_point = gs.array(
            [
                16.0,
                -2.0,
            ]
        )
        base_point = unnorm_base_point / gs.linalg.norm(unnorm_base_point)
        smoke_data = [
            dict(
                dim=1,
                vector=gs.array([0.1, 0.8]),
                base_point=base_point,
            )
        ]
        return self.generate_tests(smoke_data)


@geomstats.tests.autograd_and_torch_only
class TestHypersphereBisMetric(AbstractHypersphereMetric, metaclass=Parametrizer):
    metric = connection = CircleAsSpecialOrthogonalMetric
    skip_test_exp_geodesic_ivp = True
    skip_test_dist_point_to_itself_is_zero = True
    skip_test_christoffels_shape = True
    skip_test_sectional_curvature = True
    skip_test_triangle_inequality_of_dist = True
    skip_test_parallel_transport_bvp_is_isometry = True
    skip_test_parallel_transport_ivp_is_isometry = True

    testing_data = HypersphereBisMetricTestData()


# Then we run PullbackDiffeo test


@geomstats.tests.autograd_and_torch_only
class TestPullbackDiffeoCircle(PullbackDiffeoMetricTestCase, metaclass=Parametrizer):

    metric = CircleAsSpecialOrthogonalMetric

    testing_data = PullbackDiffeoCircleMetricTestData()
