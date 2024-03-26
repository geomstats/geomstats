import random

import pytest

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import (
    BiInvariantMetric,
    _InvariantMetricMatrix,
    _InvariantMetricVector,
)
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.random import RandomDataGenerator
from geomstats.test_cases.geometry.invariant_metric import (
    BiInvariantMetricTestCase,
    InvariantMetricMatrixTestCase,
    InvariantMetricVectorTestCase,
)

from .data.invariant_metric import (
    BiInvariantMetricMatrixTestData,
    BiInvariantMetricVectorsSOTestData,
    InvariantMetricMatrixSETestData,
    InvariantMetricMatrixSO3TestData,
    InvariantMetricMatrixSONonDiagTestData,
    InvariantMetricMatrixSOTestData,
    InvariantMetricVectorTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        (random.randint(2, 3), True, True),
        (random.randint(2, 3), True, False),
        (random.randint(2, 3), False, True),
    ],
)
def equipped_SO_matrix_groups_left_right(request):
    """SO matrix spaces to equip with invariant metric.

    Third param controls metric_matrix_at_identity:
    * True: FrobeniusMetric
    * False: random diagonal matrix
    """
    n, left, metric_mat_at_identity = request.param
    space = request.cls.space = SpecialOrthogonal(n, equip=False)

    if metric_mat_at_identity:
        metric_mat_at_identity = None
    else:
        spd_mat = SPDMatrices(space.dim, equip=False).random_point()
        metric_mat_at_identity = Matrices.to_diagonal(spd_mat)

    space.equip_with_metric(
        _InvariantMetricMatrix, left=left, metric_mat_at_identity=metric_mat_at_identity
    )

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.slow
@pytest.mark.usefixtures("equipped_SO_matrix_groups_left_right")
class TestInvariantMetricMatrixSO(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricMatrixSOTestData()


@pytest.mark.slow
class TestInvariantMetricMatrixSONonDiag(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    _n = random.randint(2, 3)
    space = SpecialOrthogonal(_n, equip=False)
    _metric_mat_at_id = SPDMatrices(space.dim, equip=False).random_point()

    space.equip_with_metric(
        _InvariantMetricMatrix, left=True, metric_mat_at_identity=_metric_mat_at_id
    )

    data_generator = RandomDataGenerator(space, amplitude=10.0)

    testing_data = InvariantMetricMatrixSONonDiagTestData()


@pytest.mark.smoke
class TestInvariantMetricMatrixSO3(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    space = SpecialOrthogonal(3, equip=False)
    space.equip_with_metric(_InvariantMetricMatrix, left=True)
    testing_data = InvariantMetricMatrixSO3TestData()

    def test_connection_translation_map(self, n_points, atol):
        # TODO: simplify code or apply to other quantities
        # e.g. sectional_curvature, curvature, curvature derivative
        base_point = self.data_generator.random_point()

        x, y, z = self.space.metric.normal_basis(self.space.lie_algebra.basis)
        expected = 1.0 / 2**0.5 / 2.0 * z

        translation_map = self.space.tangent_translation_map(base_point)
        tan_a = translation_map(x)
        tan_b = translation_map(y)

        res = self.space.metric.connection(tan_a, tan_b, base_point)
        expected = translation_map(expected)
        self.assertAllClose(res, expected, atol=atol)

    def test_connection_smoke(self, atol):
        base_point = self.space.identity
        x, y, z = self.space.metric.normal_basis(self.space.lie_algebra.basis)
        expected = 1.0 / 2**0.5 / 2.0 * z
        self.test_connection(x, y, base_point, expected, atol)

    def test_sectional_curvature_smoke(self, atol):
        base_point = self.space.identity
        x, y, z = self.space.metric.normal_basis(self.space.lie_algebra.basis)

        self.test_sectional_curvature(x, y, base_point, 1.0 / 8, atol)
        self.test_sectional_curvature(y, y, base_point, 0.0, atol)
        self.test_sectional_curvature(
            gs.stack([y, y]),
            gs.stack([z] * 2),
            base_point,
            gs.array([1.0 / 8, 1.0 / 8]),
            atol,
        )

    def test_curvature_smoke(self, atol):
        base_point = self.space.identity
        x, y, z = self.space.metric.normal_basis(self.space.lie_algebra.basis)

        self.test_curvature(x, y, x, base_point, 1.0 / 8 * y, atol)
        self.test_curvature(
            tangent_vec_a=gs.stack([x, x]),
            tangent_vec_b=gs.stack([y] * 2),
            tangent_vec_c=gs.stack([x, x]),
            base_point=base_point,
            expected=gs.array([1.0 / 8 * y] * 2),
            atol=atol,
        )
        self.test_curvature(
            tangent_vec_a=y,
            tangent_vec_b=y,
            tangent_vec_c=z,
            base_point=base_point,
            expected=gs.zeros_like(z),
            atol=atol,
        )

    def test_structure_constant_smoke(self, atol):
        # TODO: simplify by testing properties
        x, y, z = self.space.metric.normal_basis(self.space.lie_algebra.basis)

        self.test_structure_constant(
            tangent_vec_a=x,
            tangent_vec_b=y,
            tangent_vec_c=z,
            expected=2.0**0.5 / 2.0,
            atol=atol,
        )
        self.test_structure_constant(
            tangent_vec_a=y,
            tangent_vec_b=x,
            tangent_vec_c=z,
            expected=-(2.0**0.5 / 2.0),
            atol=atol,
        )
        self.test_structure_constant(
            tangent_vec_a=y,
            tangent_vec_b=z,
            tangent_vec_c=x,
            expected=2.0**0.5 / 2.0,
            atol=atol,
        )
        self.test_structure_constant(
            tangent_vec_a=z,
            tangent_vec_b=y,
            tangent_vec_c=x,
            expected=-(2.0**0.5 / 2.0),
            atol=atol,
        )
        self.test_structure_constant(
            tangent_vec_a=z,
            tangent_vec_b=x,
            tangent_vec_c=y,
            expected=2.0**0.5 / 2.0,
            atol=atol,
        )
        self.test_structure_constant(
            tangent_vec_a=x,
            tangent_vec_b=z,
            tangent_vec_c=y,
            expected=-(2.0**0.5 / 2.0),
            atol=atol,
        )
        self.test_structure_constant(
            tangent_vec_a=x,
            tangent_vec_b=x,
            tangent_vec_c=y,
            expected=0.0,
            atol=atol,
        )

    def test_curvature_derivative_at_identity_smoke(self, atol):
        x, y, z = self.space.metric.normal_basis(self.space.lie_algebra.basis)

        self.test_curvature_derivative_at_identity(
            tangent_vec_a=x,
            tangent_vec_b=y,
            tangent_vec_c=z,
            tangent_vec_d=x,
            expected=gs.zeros_like(x),
            atol=atol,
        )

    def test_inner_product_from_vec_representation(
        self, tangent_vec_a, tangent_vec_b, expected, atol
    ):
        algebra = self.space.lie_algebra
        tangent_vec_a = algebra.matrix_representation(tangent_vec_a)
        tangent_vec_b = algebra.matrix_representation(tangent_vec_b)

        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b)
        self.assertAllClose(res, expected, atol=atol)


@pytest.fixture(
    scope="class",
    params=[random.randint(2, 3)],
)
def equipped_SE_matrix_groups(request):
    n = request.param
    space = request.cls.space = SpecialEuclidean(n, equip=False)
    space.equip_with_metric(_InvariantMetricMatrix, left=False)

    request.cls.data_generator = RandomDataGenerator(space, amplitude=10.0)


@pytest.mark.slow
@pytest.mark.usefixtures("equipped_SE_matrix_groups")
class TestInvariantMetricMatrixSE(
    InvariantMetricMatrixTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricMatrixSETestData()


@pytest.fixture(
    scope="class",
    params=[
        (SpecialOrthogonal(3, point_type="vector", equip=False), True),
        (SpecialOrthogonal(3, point_type="vector", equip=False), False),
        (SpecialEuclidean(2, point_type="vector", equip=False), True),
        (SpecialEuclidean(2, point_type="vector", equip=False), False),
        (SpecialEuclidean(3, point_type="vector", equip=False), True),
        (SpecialEuclidean(3, point_type="vector", equip=False), False),
    ],
)
def equipped_vector_groups_left_right(request):
    space, left = request.param
    request.cls.space = space
    space.equip_with_metric(_InvariantMetricVector, left=left)


@pytest.mark.slow
@pytest.mark.usefixtures("equipped_vector_groups_left_right")
class TestInvariantMetricVector(
    InvariantMetricVectorTestCase, metaclass=DataBasedParametrizer
):
    testing_data = InvariantMetricVectorTestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialOrthogonal(2, point_type="vector", equip=False),
        SpecialOrthogonal(3, point_type="vector", equip=False),
    ],
)
def equipped_vector_groups(request):
    request.cls.space = space = request.param
    space.equip_with_metric(BiInvariantMetric)


@pytest.mark.usefixtures("equipped_vector_groups")
class TestBiInvariantMetricVectorsSO(
    BiInvariantMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BiInvariantMetricVectorsSOTestData()


@pytest.fixture(
    scope="class",
    params=[
        SpecialOrthogonal(3, equip=False),
    ],
)
def equipped_SO_matrix_groups(request):
    request.cls.space = space = request.param
    space.equip_with_metric(BiInvariantMetric)


@pytest.mark.usefixtures("equipped_SO_matrix_groups")
class TestBiInvariantMetricMatrixSO(
    BiInvariantMetricTestCase, metaclass=DataBasedParametrizer
):
    testing_data = BiInvariantMetricMatrixTestData()
