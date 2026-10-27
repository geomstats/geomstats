import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import VectorSpaceTestData
from .riemannian_metric import RiemannianMetricTestData
from .mixins import GroupExpMixinsTestData


class HermitianTestData(GroupExpMixinsTestData, VectorSpaceTestData):
    def exp_random_test_data(self):
        return self.generate_random_data()

    def identity_belongs_test_data(self):
        return self.generate_tests([dict()])


class HermitianMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class Hermitian2MetricTestData(TestData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([0.0 + 0.0j, 1.0 + 1.0j]),
                base_point=gs.array([2.0 + 2.0j, 10.0 + 10.0j]),
                expected=gs.array([2.0 + 2.0j, 11.0 + 11.0j]),
            ),
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([2.0 + 2.0j, 10.0 + 10.0j]),
                base_point=gs.array([0.0 + 0.0j, 1.0 + 1.0j]),
                expected=gs.array([2.0 + 2.0j, 9.0 + 9.0j]),
            ),
        ]
        return self.generate_tests(data)

    def metric_matrix_test_data(self):
        data = [
            dict(base_point=None, expected=gs.eye(2, dtype=gs.get_default_cdtype())),
        ]
        return self.generate_tests(data)
