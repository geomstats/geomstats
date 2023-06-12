from tests2.tests_geomstats.test_geometry.data.base import OpenSetTestData
from tests2.tests_geomstats.test_geometry.data.manifold import ManifoldTestData
from tests2.tests_geomstats.test_geometry.data.poincare_half_space import (
    PoincareHalfSpaceTestData,
)
from tests2.tests_geomstats.test_geometry.data.pullback_metric import (
    PullbackDiffeoMetricTestData,
)
from tests2.tests_geomstats.test_geometry.data.riemannian_metric import (
    RiemannianMetricTestData,
)
from tests2.tests_geomstats.test_geometry.data.spd_matrices import (
    SPDAffineMetricTestData,
    SPDMatricesTestData,
)
from tests2.tests_geomstats.test_information_geometry.data.base import (
    InformationManifoldMixinTestData,
)


class UnivariateNormalDistributionsTestData(
    InformationManifoldMixinTestData, PoincareHalfSpaceTestData
):
    pass


class UnivariateNormalMetricTestData(PullbackDiffeoMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class CenteredNormalDistributionsTestData(
    InformationManifoldMixinTestData, SPDMatricesTestData
):
    pass


class CenteredNormalMetricTestData(SPDAffineMetricTestData):
    pass


class DiagonalNormalDistributionsTestData(
    InformationManifoldMixinTestData, OpenSetTestData
):
    pass


class DiagonalNormalMetricTestData(RiemannianMetricTestData):
    trials = 2
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    tolerances = {"geodesic_ivp_belongs": {"atol": 1e-3}}


class GeneralNormalDistributionsTestData(
    InformationManifoldMixinTestData, ManifoldTestData
):
    skips = ("not_belongs",)


class GeneralNormalMetricTestData(RiemannianMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False
