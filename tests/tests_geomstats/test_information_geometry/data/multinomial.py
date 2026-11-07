from polpo.testing.data import CaseData

import geomstats.backend as gs

from ...test_geometry.data.base import LevelSetTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class MultinomialDistributionsTestData(
    InformationManifoldMixinTestData, LevelSetTestData
):
    pass


class MultinomialDistributions2TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([-1.0, 0.3]), expected=False),
        ]
        return data


class MultinomialDistributions3TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([0.1, 0.2, 0.3, 0.4]), expected=True),
            dict(point=gs.array([0.0, 1.0, 0.3, 0.4]), expected=False),
        ]
        return data


class MultinomialMetricTestData(RiemannianMetricTestData):
    def sectional_curvature_is_positive_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    def dist_against_closed_form_test_data(self):
        return self.generate_random_data(arg_names=("point_a", "point_b"))
