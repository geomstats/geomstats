from polpo.testing.data import CaseData

import geomstats.backend as gs

from .riemannian_metric import RiemannianMetricTestData


class Hyperboloid2TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array([0.5, 7, 3.0]),
                expected=False,
            ),
        ]
        return data


class Hyperboloid3TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array([1.0, 0.0, 0.0, 0.0]),
                expected=True,
            ),
        ]
        return data


class HyperboloidMetricTestData(RiemannianMetricTestData):
    def inner_product_is_minkowski_inner_product_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )
