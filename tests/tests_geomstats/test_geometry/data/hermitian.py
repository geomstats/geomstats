from polpo.testing.data import CaseData

import geomstats.backend as gs

from .base import ComplexVectorSpaceTestData
from .mixins import GroupExpMixinsTestData


class Hermitian2MetricTestData(CaseData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([0.0 + 0.0j, 1.0 + 1.0j]),
                base_point=gs.array([2.0 + 2.0j, 10.0 + 10.0j]),
                expected=gs.array([2.0 + 2.0j, 11.0 + 11.0j]),
            ),
        ]
        return data

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([2.0 + 2.0j, 10.0 + 10.0j]),
                base_point=gs.array([0.0 + 0.0j, 1.0 + 1.0j]),
                expected=gs.array([2.0 + 2.0j, 9.0 + 9.0j]),
            ),
        ]
        return data

    def metric_matrix_test_data(self):
        data = [
            dict(base_point=None, expected=gs.eye(2, dtype=gs.get_default_cdtype())),
        ]
        return data


class HermitianTestData(GroupExpMixinsTestData, ComplexVectorSpaceTestData):
    def exp_random_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def identity_belongs_test_data(self):
        return [dict()]
