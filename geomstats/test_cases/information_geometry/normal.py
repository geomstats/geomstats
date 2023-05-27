from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.poincare_half_space import PoincareHalfSpaceTestCase
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.geometry.spd_matrices import (
    SPDAffineMetricTestCase,
    SPDMatricesTestCase,
)
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class UnivariateNormalDistributionsTestCase(
    InformationManifoldMixinTestCase, PoincareHalfSpaceTestCase
):
    pass


class UnivariateNormalMetricTestCase(PullbackDiffeoMetricTestCase):
    pass


class CenteredNormalDistributionsTestCase(
    InformationManifoldMixinTestCase, SPDMatricesTestCase
):
    pass


class CenteredNormalMetricTestCase(SPDAffineMetricTestCase):
    pass


class DiagonalNormalDistributionsTestCase(
    InformationManifoldMixinTestCase, OpenSetTestCase
):
    pass


class DiagonalNormalMetricTestCase(RiemannianMetricTestCase):
    pass


class GeneralNormalDistributionsTestCase(
    InformationManifoldMixinTestCase, ManifoldTestCase
):
    # TODO: inherit from ManifoldTestCase?
    pass


class GeneralNormalMetricTestCase(RiemannianMetricTestCase):
    # TODO: inherit from ProductMetricTestCase?
    pass
