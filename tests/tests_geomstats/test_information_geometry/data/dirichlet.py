from polpo.testing.data import CaseData

import geomstats.backend as gs

from ...test_geometry.data.base import VectorSpaceOpenSetTestData
from ...test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .base import InformationManifoldMixinTestData


class DirichletDistributionsTestData(
    InformationManifoldMixinTestData, VectorSpaceOpenSetTestData
):
    pass


class DirichletDistributions3TestData(CaseData):
    def belongs_test_data(self):
        smoke_data = [
            dict(point=gs.array([0.1, 1.0, 0.3]), expected=True),
            dict(point=gs.array([0.1, 1.001]), expected=False),
            dict(point=gs.array([-0.001, 1.0, 0.3]), expected=False),
        ]
        return smoke_data


class DirichletMetricTestData(RiemannianMetricTestData):
    def jacobian_christoffels_vec_test_data(self):
        return self.generate_vectorization_data(arg_names="base_point", on_metric=True)

    def sectional_curvature_is_negative_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b")
        )

    def exp_diagonal_is_totally_geodesic_test_data(self):
        return [dict(param=gs.random.rand(1), tangent_param=gs.random.rand(1))]
