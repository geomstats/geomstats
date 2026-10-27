import random

import pytest
from polpo.testing.parametrizers import (
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.diffeo import ComposedDiffeo
from geomstats.geometry.full_rank_correlation_matrices import (
    EuclideanCholeskyDiffeo,
    EuclideanCholeskyMetric,
    FullRankCorrelationMatrices,
    LogEuclideanCholeskyDiffeo,
    LogEuclideanCholeskyMetric,
    LogScaledMetric,
    LogScalingDiffeo,
    OffLogDiffeo,
    OffLogMetric,
    PolyHyperbolicCholeskyMetric,
    SPDScalingFinder,
    UniqueDiagonalMatrixAlgorithm,
)
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.hermitian_matrices import expmh
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.lower_triangular_matrices import StrictlyLowerTriangularMatrices
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.open_hemisphere import (
    OpenHemispheresProduct,
    OpenHemisphereToHyperboloidDiffeo,
)
from geomstats.geometry.positive_lower_triangular_matrices import (
    PLTUnitDiagMatrices,
    UnitNormedRowsPLTDiffeo,
    UnitNormedRowsPLTMatrices,
)
from geomstats.geometry.spd_matrices import CholeskyMap, SPDMatrices
from geomstats.geometry.symmetric_matrices import (
    NullRowSumsSymmetricMatrices,
    SymmetricHollowMatrices,
    SymmetricMatrices,
)
from geomstats.numerics.optimization import NewtonMethod
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.full_rank_correlation_matrices import (
    FullRankCorrelationMatricesTestCase,
)
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.diffeo import DiffeoTestData
from .data.full_rank_correlation_matrices import (
    SPDScalingFinderTestData,
    UniqueDiagonalMatrixAlgorithmTestData,
)
from .data.riemannian_metric import RiemannianMetricTestData

# from .data.full_rank_correlation_matrices import (
#     CorrelationMatricesBundleTestData,
#     EuclideanCholeskyMetricTestData,
#     FullRankCorrelationAffineQuotientMetricTestData,
#     FullRankCorrelationMatricesTestData,
#     LogScaledMetricTestData,
#     OffLogMetricTestData,
#     PolyHyperbolicCholeskyMetricTestData,
#     SPDScalingFinderTestData,
#     UniqueDiagonalMatrixAlgorithmTestData,
# )


# @pytest.fixture(
#     scope="class",
#     params=[
#         3,
#         random.randint(4, 8),
#     ],
# )
# def spaces(request):
#     request.cls.space = FullRankCorrelationMatrices(n=request.param, equip=False)


# @pytest.mark.usefixtures("spaces")
# class TestFullRankCorrelationMatrices(
#     FullRankCorrelationMatricesTestCase, metaclass=DataBasedParametrizer
# ):
#     testing_data = FullRankCorrelationMatricesTestData()


# @pytest.fixture(
#     scope="class",
#     params=[
#         2,
#         random.randint(3, 5),
#     ],
# )
# def bundles(request):
#     n = request.param
#     request.cls.total_space = total_space = SPDMatrices(n=n, equip=True)

#     total_space.equip_with_group_action(FullRankCorrelationMatrices.diag_action)
#     total_space.equip_with_quotient()

#     request.cls.base = FullRankCorrelationMatrices(n=n, equip=False)


# @pytest.mark.usefixtures("bundles")
# class TestCorrelationMatricesBundle(
#     FiberBundleTestCase, metaclass=DataBasedParametrizer
# ):
#     testing_data = CorrelationMatricesBundleTestData()

#     def test_horizontal_projection_is_horizontal_v2(self, n_points, atol):
#         base_point = self.data_generator.random_point(n_points)
#         tangent_vec = self.data_generator.random_tangent_vec(base_point)

#         horizontal_vec = self.total_space.fiber_bundle.horizontal_projection(
#             tangent_vec, base_point
#         )

#         inverse = GeneralLinear.inverse(base_point)
#         product_1 = Matrices.mul(horizontal_vec, inverse)
#         product_2 = Matrices.mul(inverse, horizontal_vec)
#         is_horizontal = gs.all(
#             self.base.is_tangent(product_1 + product_2, base_point, atol=atol)
#         )

#         self.assertTrue(is_horizontal)


# @pytest.mark.redundant
# class TestFullRankCorrelationAffineQuotientMetric(
#     QuotientMetricTestCase, metaclass=DataBasedParametrizer
# ):
#     _n = random.randint(3, 5)
#     space = FullRankCorrelationMatrices(n=_n)
#     testing_data = FullRankCorrelationAffineQuotientMetricTestData()


class TestCholeskyMap(
    DiffeoTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 5)

    domain_space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = UnitNormedRowsPLTMatrices(n=_n, equip=False)
    diffeo = CholeskyMap()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestDiffeoToOpenHemispheres(
    DiffeoTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(3, 5)

    domain_space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = OpenHemispheresProduct(n=_n, equip=False)

    _diffeos = [CholeskyMap(), UnitNormedRowsPLTDiffeo(_n)]
    diffeo = ComposedDiffeo(_diffeos)

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestDiffeoToHyperboloid(
    DiffeoTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = 2

    domain_space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = Hyperboloid(dim=1, equip=False)

    _diffeos = [
        CholeskyMap(),
        UnitNormedRowsPLTDiffeo(_n),
        OpenHemisphereToHyperboloidDiffeo(),
    ]
    diffeo = ComposedDiffeo(_diffeos)

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def phc_equipped_spaces(request):
    n = request.param
    space = request.cls.space = FullRankCorrelationMatrices(
        n, equip=False
    ).equip_with_metric(
        PolyHyperbolicCholeskyMetric,
    )

    request.cls.testing_data.space = space


@pytest.mark.redundant
@pytest.mark.usefixtures("phc_equipped_spaces")
class TestPolyHyperbolicCholeskyMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = RiemannianMetricTestData()


class TestEuclideanCholeskyDiffeo(
    DiffeoTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 5)

    domain_space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = PLTUnitDiagMatrices(n=_n, equip=False)
    diffeo = EuclideanCholeskyDiffeo()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.mark.redundant
class TestEuclideanCholeskyMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)

    space = FullRankCorrelationMatrices(n=_n, equip=False).equip_with_metric(
        EuclideanCholeskyMetric
    )

    testing_data = RiemannianMetricTestData(space)


class TestLogEuclideanCholeskyDiffeo(
    DiffeoTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 5)

    domain_space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = StrictlyLowerTriangularMatrices(n=_n, equip=False)
    diffeo = LogEuclideanCholeskyDiffeo()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.mark.redundant
class TestLogEuclideanCholeskyMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)

    space = FullRankCorrelationMatrices(n=_n, equip=False).equip_with_metric(
        LogEuclideanCholeskyMetric
    )
    testing_data = RiemannianMetricTestData(space)


class TestUniqueDiagonalMatrixAlgorithm(
    TestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)

    domain_space = SymmetricMatrices(n=_n, equip=False)
    image_space = FullRankCorrelationMatrices(n=_n, equip=False)

    map = staticmethod(
        lambda point: expmh(UniqueDiagonalMatrixAlgorithm()(point) + point)
    )

    testing_data = UniqueDiagonalMatrixAlgorithmTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=map,
    )

    def test_map(self, point, expected, atol=gs.atol):
        self.assertAllClose(self.map(point), expected, atol=atol)

    def test_map_belongs_to_image(self, point, atol=gs.atol):
        image_point = self.map(point)

        res = self.image_space.belongs(image_point, atol=atol)
        expected = gs.ones_like(res)

        self.assertAllEqual(res, expected)


class TestOffLogDiffeo(
    DiffeoTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 5)

    domain_space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = SymmetricHollowMatrices(n=_n, equip=False)

    diffeo = OffLogDiffeo()
    testing_data = DiffeoTestData()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.fixture(
    scope="class",
    params=[
        (2, (0.0, 0.0, 1.0)),
        (3, (0.0, 1.0, 1.0)),
        (random.randint(4, 5), (1.0, 1.0, 1.0)),
    ],
)
def equipped_cor_with_off_log_metric(request):
    n, (alpha, beta, gamma) = request.param
    space = request.cls.space = FullRankCorrelationMatrices(
        n, equip=False
    ).equip_with_metric(OffLogMetric, alpha=alpha, beta=beta, gamma=gamma)

    request.cls.testing_data.space = space


@pytest.mark.redundant
@pytest.mark.usefixtures("equipped_cor_with_off_log_metric")
class TestOffLogMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = RiemannianMetricTestData()


@pytest.fixture(
    scope="class",
    params=[
        NewtonMethod(damped=True),
    ],
)
def unique_positive_diagonal_matrix_algorithms(request):
    root_finder = request.param
    map_ = request.cls.map = SPDScalingFinder(root_finder)

    request.cls.testing_data.map = lambda point: map_(point)


@pytest.mark.usefixtures("unique_positive_diagonal_matrix_algorithms")
class TestSPDScalingFinder(TestCase, metaclass=GeometricDataBasedParametrizer):
    _n = random.randint(2, 5)

    domain_space = SPDMatrices(n=_n, equip=False)

    testing_data = SPDScalingFinderTestData(
        domain_space=domain_space,
    )

    def test_map(self, point, expected, atol=gs.atol):
        self.assertAllClose(self.map(point), expected, atol=atol)

    def test_rows_sum_to_one(self, point, atol=gs.atol):
        diag_vec = self.map(point)

        unit_row_sum_spd = point * gs.outer(diag_vec, diag_vec)

        res = gs.sum(unit_row_sum_spd, axis=-1)
        expected = gs.ones_like(res)

        self.assertAllClose(res, expected, atol=atol)

    def test_values_are_positive(self, point):
        diag_vec = self.map(point)

        self.assertAllEqual(
            diag_vec > 0.0,
            gs.ones_like(diag_vec, dtype=bool),
        )


class TestLogScalingDiffeo(
    DiffeoTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    _n = random.randint(2, 5)

    domain_space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = NullRowSumsSymmetricMatrices(n=_n, equip=False)
    diffeo = LogScalingDiffeo()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


@pytest.fixture(
    scope="class",
    params=[
        (2, (0.0, 0.0, 1.0)),
        (3, (0.0, 1.0, 1.0)),
        (random.randint(4, 5), (1.0, 1.0, 1.0)),
    ],
)
def equipped_cor_with_log_scaled_metric(request):
    n, (alpha, delta, zeta) = request.param
    space = request.cls.space = FullRankCorrelationMatrices(
        n, equip=False
    ).equip_with_metric(LogScaledMetric, alpha=alpha, delta=delta, zeta=zeta)

    request.cls.testing_data.space = space


@pytest.mark.redundant
@pytest.mark.usefixtures("equipped_cor_with_log_scaled_metric")
class TestLogScaledMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    testing_data = RiemannianMetricTestData()
