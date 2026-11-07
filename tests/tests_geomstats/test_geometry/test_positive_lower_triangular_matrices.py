import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.lower_triangular_matrices import StrictlyLowerTriangularMatrices
from geomstats.geometry.positive_lower_triangular_matrices import (
    CholeskyMetric,
    InvariantPositiveLowerTriangularMatricesMetric,
    LowerMatrixLog,
    PLTUnitDiagMatrices,
    PositiveLowerTriangularMatrices,
    UnitNormedRowsPLTDiffeo,
    UnitNormedRowsPLTMatrices,
)
from geomstats.test_cases.geometry.base import (
    DiffeomorphicManifoldTestCase,
    LevelSetTestCase,
    VectorSpaceOpenSetTestCase,
)
from geomstats.test_cases.geometry.diffeo import (
    DiffeoComparisonTestCase,
    DiffeoTestCase,
)
from geomstats.test_cases.geometry.invariant_metric import InvariantMetricMatrixTestCase
from geomstats.test_cases.geometry.lie_group import MatrixLieGroupTestCase
from geomstats.test_cases.geometry.positive_lower_triangular_matrices import (
    CholeskyMetricTestCase,
)
from geomstats.test_cases.geometry.pullback_metric import PullbackDiffeoMetricTestCase
from geomstats.variants.geometry.positive_lower_triangular_matrices import (
    LowerMatrixLog as OtherLowerMatrixLog,
)

from .data.base import DiffeomorphicManifoldTestData, LevelSetTestData
from .data.diffeo import DiffeoComparisonTestData, DiffeoTestData
from .data.invariant_metric import InvariantMetricMatrixTestData
from .data.positive_lower_triangular_matrices import (
    CholeskyMetric2TestData,
    CholeskyMetricTestData,
    PositiveLowerTriangularMatrices2TestData,
    PositiveLowerTriangularMatricesTestData,
)
from .data.riemannian_metric import RiemannianMetricTestData


@pytest.mark.smoke
class TestPositiveLowerTriangularMatrices2(
    MatrixLieGroupTestCase, VectorSpaceOpenSetTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveLowerTriangularMatrices(n=2, equip=False)
    testing_data = PositiveLowerTriangularMatrices2TestData()


class TestPositiveLowerTriangularMatrices(
    MatrixLieGroupTestCase,
    VectorSpaceOpenSetTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = PositiveLowerTriangularMatrices(n=random.randint(2, 5), equip=False)
    testing_data = PositiveLowerTriangularMatricesTestData(
        space,
        # NB: the inverse of base points near the boundary is ill-conditioned
        method_marks={"xfail": ("log_after_exp",)},
    )


@pytest.mark.smoke
class TestCholeskyMetric2(CholeskyMetricTestCase, metaclass=DataBasedParametrizer):
    space = PositiveLowerTriangularMatrices(n=2, equip=False)
    space.equip_with_metric(CholeskyMetric)
    testing_data = CholeskyMetric2TestData()


class TestCholeskyMetric(
    CholeskyMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PositiveLowerTriangularMatrices(n=random.randint(2, 5))
    testing_data = CholeskyMetricTestData(space)


# NB: `DataBasedParametrizer` skips vectorization tests, which are slow here
@pytest.mark.slow
@pytest.mark.redundant
class TestInvariantPositiveLowerTriangularMatricesMetric(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    space = PositiveLowerTriangularMatrices(n=2, equip=False).equip_with_metric(
        InvariantPositiveLowerTriangularMatricesMetric
    )
    testing_data = InvariantMetricMatrixTestData(
        space,
        excluded_methods=("parallel_transport_ivp_norm",),
        method_marks={
            "xfail": ("exp_after_log", "log_after_exp", "parallel_transport_bvp_norm")
        },
    )


class TestUnitNormedRowsPLTDiffeo(
    DiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)
    domain_space = UnitNormedRowsPLTMatrices(n=_n, equip=False)
    image_space = domain_space.image_space
    diffeo = UnitNormedRowsPLTDiffeo(_n)

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestUnitNormedRowsPLTMatrices(
    DiffeomorphicManifoldTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = UnitNormedRowsPLTMatrices(n=random.randint(2, 5), equip=False)
    testing_data = DiffeomorphicManifoldTestData(space)


class TestUnitNormedRowsPLTMatricesPullbackMetric(
    PullbackDiffeoMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = UnitNormedRowsPLTMatrices(n=random.randint(2, 5))
    testing_data = RiemannianMetricTestData(space, generator_kwargs={"amplitude": 5.0})


class TestPLTUnitDiagMatrices(
    LevelSetTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = PLTUnitDiagMatrices(n=random.randint(2, 5), equip=False)
    testing_data = LevelSetTestData(space)


class TestLowerMatrixLog(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _n = random.randint(2, 5)
    domain_space = PLTUnitDiagMatrices(n=_n, equip=False)
    image_space = StrictlyLowerTriangularMatrices(n=_n, equip=False)
    diffeo = LowerMatrixLog()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestLowerMatrixLogCmp(
    DiffeoComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _n = random.randint(2, 5)
    domain_space = PLTUnitDiagMatrices(n=_n, equip=False)
    image_space = StrictlyLowerTriangularMatrices(n=_n, equip=False)
    diffeo = LowerMatrixLog()
    other_diffeo = OtherLowerMatrixLog()

    testing_data = DiffeoComparisonTestData(
        domain_space=domain_space,
        image_space=image_space,
    )
