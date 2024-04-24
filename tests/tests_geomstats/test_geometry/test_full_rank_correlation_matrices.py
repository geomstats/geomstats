import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.diffeo import ComposedDiffeo
from geomstats.geometry.full_rank_correlation_matrices import (
    FullRankCorrelationMatrices,
    LogScaledMetric,
    LogScalingDiffeo,
    OffLogDiffeo,
    OffLogMetric,
    PolyHyperbolicCholeskyMetric,
    UniqueDiagonalMatrixAlgorithm,
    UniquePositiveDiagonalMatrixAlgorithm,
)
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.hermitian_matrices import expmh
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.open_hemisphere import (
    OpenHemispheresProduct,
    OpenHemisphereToHyperboloidDiffeo,
)
from geomstats.geometry.positive_lower_triangular_matrices import (
    UnitNormedRowsPLTDiffeo,
    UnitNormedRowsPLTMatrices,
)
from geomstats.geometry.spd_matrices import CholeskyMap, SPDMatrices
from geomstats.geometry.symmetric_matrices import (
    NullRowSumsSymmetricMatrices,
    SymmetricHollowMatrices,
    SymmetricMatrices,
)
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test.test_case import TestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.fiber_bundle import FiberBundleTestCase
from geomstats.test_cases.geometry.full_rank_correlation_matrices import (
    FullRankCorrelationMatricesTestCase,
)
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.test_cases.geometry.quotient_metric import QuotientMetricTestCase

from .data.diffeo import DiffeoTestData
from .data.full_rank_correlation_matrices import (
    CorrelationMatricesBundleTestData,
    FullRankCorrelationAffineQuotientMetricTestData,
    FullRankCorrelationMatricesTestData,
    LogScaledMetricTestData,
    OffLogMetricTestData,
    PolyHyperbolicCholeskyMetricTestData,
    UniqueDiagonalMatrixAlgorithmTestData,
    UniquePositiveDiagonalMatrixAlgorithmTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        3,
        random.randint(4, 8),
    ],
)
def spaces(request):
    request.cls.space = FullRankCorrelationMatrices(n=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestFullRankCorrelationMatrices(
    FullRankCorrelationMatricesTestCase, metaclass=DataBasedParametrizer
):
    testing_data = FullRankCorrelationMatricesTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def bundles(request):
    n = request.param
    request.cls.total_space = total_space = SPDMatrices(n=n, equip=True)

    total_space.equip_with_group_action(FullRankCorrelationMatrices.diag_action)
    total_space.equip_with_quotient_structure()

    request.cls.base = FullRankCorrelationMatrices(n=n, equip=False)


@pytest.mark.usefixtures("bundles")
class TestCorrelationMatricesBundle(
    FiberBundleTestCase, metaclass=DataBasedParametrizer
):
    testing_data = CorrelationMatricesBundleTestData()

    def test_horizontal_projection_is_horizontal_v2(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        horizontal_vec = self.total_space.fiber_bundle.horizontal_projection(
            tangent_vec, base_point
        )

        inverse = GeneralLinear.inverse(base_point)
        product_1 = Matrices.mul(horizontal_vec, inverse)
        product_2 = Matrices.mul(inverse, horizontal_vec)
        is_horizontal = gs.all(
            self.base.is_tangent(product_1 + product_2, base_point, atol=atol)
        )

        self.assertTrue(is_horizontal)


@pytest.mark.redundant
class TestFullRankCorrelationAffineQuotientMetric(
    QuotientMetricTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(3, 5)
    space = FullRankCorrelationMatrices(n=_n)
    testing_data = FullRankCorrelationAffineQuotientMetricTestData()


class TestCholeskyMap(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = UnitNormedRowsPLTMatrices(n=_n, equip=False)
    diffeo = CholeskyMap()
    testing_data = DiffeoTestData()


class TestDiffeoToOpenHemispheres(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(3, 5)
    space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = OpenHemispheresProduct(n=_n, equip=False)
    _diffeos = [CholeskyMap(), UnitNormedRowsPLTDiffeo(_n)]
    diffeo = ComposedDiffeo(_diffeos)
    testing_data = DiffeoTestData()


class TestDiffeoToHyperboloid(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = 2
    space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = Hyperboloid(dim=1, equip=False)
    _diffeos = [
        CholeskyMap(),
        UnitNormedRowsPLTDiffeo(_n),
        OpenHemisphereToHyperboloidDiffeo(),
    ]
    diffeo = ComposedDiffeo(_diffeos)
    testing_data = DiffeoTestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def phc_equipped_spaces(request):
    n = request.param
    request.cls.space = FullRankCorrelationMatrices(n, equip=False).equip_with_metric(
        PolyHyperbolicCholeskyMetric,
    )


@pytest.mark.usefixtures("phc_equipped_spaces")
class TestPolyHyperbolicCholeskyMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = PolyHyperbolicCholeskyMetricTestData()


class TestUniqueDiagonalMatrixAlgorithm(TestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    algo = UniqueDiagonalMatrixAlgorithm()
    sym_data_generator = RandomDataGenerator(SymmetricMatrices(n=_n, equip=False))
    full_rank_cor = FullRankCorrelationMatrices(n=_n, equip=False)
    testing_data = UniqueDiagonalMatrixAlgorithmTestData()

    @pytest.mark.random
    def test_belongs_to_full_rank_cor(self, n_points, atol):
        sym_mat = self.sym_data_generator.random_point(n_points)

        diag_mat = self.algo.apply(sym_mat)
        cor_mat = expmh(diag_mat + sym_mat)
        res = self.full_rank_cor.belongs(cor_mat, atol=atol)
        expected = gs.ones_like(res)
        self.assertAllEqual(res, expected)


class TestOffLogDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)
    space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = SymmetricHollowMatrices(n=_n, equip=False)
    diffeo = OffLogDiffeo()
    testing_data = DiffeoTestData()


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
    request.cls.space = FullRankCorrelationMatrices(n, equip=False).equip_with_metric(
        OffLogMetric, alpha=alpha, beta=beta, gamma=gamma
    )


@pytest.mark.usefixtures("equipped_cor_with_off_log_metric")
class TestOffLogMetric(PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = OffLogMetricTestData()


class TestUniquePositiveDiagonalMatrixAlgorithm(
    TestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 5)
    algo = UniquePositiveDiagonalMatrixAlgorithm()
    data_generator = RandomDataGenerator(SPDMatrices(n=_n, equip=False))

    testing_data = UniquePositiveDiagonalMatrixAlgorithmTestData()

    @pytest.mark.random
    def test_rows_sum_to_one(self, n_points, atol):
        spd_mat = self.data_generator.random_point(n_points)
        diag_vec = self.algo.apply(spd_mat)

        unit_row_sum_spd = spd_mat * gs.outer(diag_vec, diag_vec)

        res = gs.sum(unit_row_sum_spd, axis=-1)

        batch_shape = (n_points,) if n_points > 1 else ()
        expected = gs.ones(batch_shape + (spd_mat.shape[-1],))
        self.assertAllClose(res, expected, atol=atol)


class TestLogScalingDiffeo(DiffeoTestCase, metaclass=DataBasedParametrizer):
    _n = random.randint(2, 5)

    space = FullRankCorrelationMatrices(n=_n, equip=False)
    image_space = NullRowSumsSymmetricMatrices(n=_n, equip=False)
    diffeo = LogScalingDiffeo()
    testing_data = DiffeoTestData()


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
    request.cls.space = FullRankCorrelationMatrices(n, equip=False).equip_with_metric(
        LogScaledMetric, alpha=alpha, delta=delta, zeta=zeta
    )


@pytest.mark.usefixtures("equipped_cor_with_log_scaled_metric")
class TestLogScaledMetric(
    PullbackDiffeoMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = LogScaledMetricTestData()
