import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import VectorSpaceTestData
from .mixins import GroupExpMixinsTestData
from .riemannian_metric import RiemannianMetricTestData


class EuclideanTestData(GroupExpMixinsTestData, VectorSpaceTestData):
    def exp_random_test_data(self):
        return self.generate_random_data()

    def identity_belongs_test_data(self):
        return self.generate_tests([dict()])


class EuclideanMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def inner_product_derivative_matrix_is_zeros_test_data(self):
        return self.generate_random_data()

    def christoffels_are_zeros_test_data(self):
        return self.generate_random_data()


class CanonicalEuclideanMetricTestData(EuclideanMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    def cometrix_matrix_is_identity_test_data(self):
        return self.generate_random_data()


class CanonicalEuclideanMetric2TestData(TestData):
    def exp_test_data(self):
        tangent_vec = gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]])
        base_point = gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]])
        data = [
            dict(
                tangent_vec=tangent_vec,
                base_point=base_point,
                expected=base_point + tangent_vec,
            )
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        point = gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]])
        base_point = gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]])

        data = [
            dict(
                point=point,
                base_point=base_point,
                expected=point - base_point,
            )
        ]
        return self.generate_tests(data)

    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]]),
                tangent_vec_b=gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]]),
                base_point=None,
                expected=gs.array([14.0, -12.0, 21.0]),
            )
        ]
        return self.generate_tests(data)

    def inner_coproduct_test_data(self):
        data = [
            dict(
                cotangent_vec_a=gs.array([1.0, 2.0]),
                cotangent_vec_b=gs.array([1.0, 2.0]),
                base_point=None,
                expected=gs.array(5.0),
            ),
        ]
        return self.generate_tests(data)

    def hamiltonian_test_data(self):
        data = [
            dict(
                state=(gs.array([1.0, 2.0]), gs.array([1.0, 2.0])),
                expected=gs.array(2.5),
            )
        ]
        return self.generate_tests(data)

    def squared_norm_test_data(self):
        data = [
            dict(
                vector=gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]]),
                base_point=None,
                expected=gs.array([5.0, 20.0, 26.0]),
            )
        ]

        return self.generate_tests(data)

    def norm_test_data(self):
        data = [
            dict(
                vector=gs.array([4.0, 3.0]),
                base_point=None,
                expected=gs.array(5.0),
            )
        ]
        return self.generate_tests(data)

    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=None,
                expected=gs.eye(2),
            )
        ]
        return self.generate_tests(data)

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]]),
                point_b=gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]]),
                expected=gs.array([81.0, 109.0, 29.0]),
            )
        ]
        return self.generate_tests(data)

    def dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]]),
                point_b=gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]]),
                expected=gs.sqrt(gs.array([81.0, 109.0, 29.0])),
            )
        ]
        return self.generate_tests(data)
