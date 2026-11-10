from polpo.testing.data import CaseData

import geomstats.backend as gs

from .base import VectorSpaceTestData
from .mixins import GroupExpMixinsTestData
from .riemannian_metric import RiemannianMetricTestData


class EuclideanTestData(GroupExpMixinsTestData, VectorSpaceTestData):
    def exp_is_addition_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def identity_belongs_test_data(self):
        return [dict()]


class EuclideanMetricTestData(RiemannianMetricTestData):
    def inner_product_derivative_matrix_is_zeros_test_data(self):
        return self.generate_random_data(arg_names="base_point")

    def christoffels_are_zeros_test_data(self):
        return self.generate_random_data(arg_names="base_point")


class CanonicalEuclideanMetricTestData(EuclideanMetricTestData):
    def cometric_matrix_is_identity_test_data(self):
        return self.generate_random_data(arg_names="base_point")


class CanonicalEuclideanMetric2TestData(CaseData):
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
        return data

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
        return data

    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]]),
                tangent_vec_b=gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]]),
                base_point=None,
                expected=gs.array([14.0, -12.0, 21.0]),
            )
        ]
        return data

    def inner_coproduct_test_data(self):
        data = [
            dict(
                cotangent_vec_a=gs.array([1.0, 2.0]),
                cotangent_vec_b=gs.array([1.0, 2.0]),
                base_point=None,
                expected=gs.array(5.0),
            ),
        ]
        return data

    def hamiltonian_test_data(self):
        data = [
            dict(
                state=(gs.array([1.0, 2.0]), gs.array([1.0, 2.0])),
                expected=gs.array(2.5),
            )
        ]
        return data

    def squared_norm_test_data(self):
        data = [
            dict(
                vector=gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]]),
                base_point=None,
                expected=gs.array([5.0, 20.0, 26.0]),
            )
        ]

        return data

    def norm_test_data(self):
        data = [
            dict(
                vector=gs.array([4.0, 3.0]),
                base_point=None,
                expected=gs.array(5.0),
            )
        ]
        return data

    def metric_matrix_test_data(self):
        data = [
            dict(
                base_point=None,
                expected=gs.eye(2),
            )
        ]
        return data

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[2.0, 1.0], [-2.0, -4.0], [-5.0, 1.0]]),
                point_b=gs.array([[2.0, 10.0], [8.0, -1.0], [-3.0, 6.0]]),
                expected=gs.array([81.0, 109.0, 29.0]),
            )
        ]
        return data
