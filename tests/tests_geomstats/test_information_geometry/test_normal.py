import random

import pytest
from polpo.testing.parametrizers import GeometricDataBasedParametrizer

from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.information_geometry.normal import (
    DiagonalNormalDistributionsRandomVariable,
    MultivariateNormalDistributionsRandomVariable,
    NormalDistributions,
    SharedMeanNormalDistributionsRandomVariable,
    UnivariateNormalDistributionsRandomVariable,
    UnivariateNormalToPoincareHalfSpaceDiffeo,
)
from geomstats.test.test_case import autodiff_backend
from geomstats.test_cases.geometry.base import VectorSpaceOpenSetTestCase
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.product_manifold import ProductManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)
from geomstats.test_cases.information_geometry.normal import (
    UnivariateNormalMetricTestCase,
)

from ..test_geometry.data.diffeo import DiffeoTestData
from ..test_geometry.data.riemannian_metric import RiemannianMetricTestData
from ..test_geometry.data.spd_matrices import SPDAffineMetricTestData
from .data.normal import (
    CenteredNormalDistributionsTestData,
    DiagonalNormalDistributionsTestData,
    GeneralNormalDistributionsTestData,
    UnivariateNormalDistributionsTestData,
    UnivariateNormalMetricTestData,
)


class TestUnivariateNormalDistributions(
    InformationManifoldMixinTestCase,
    VectorSpaceOpenSetTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = NormalDistributions(sample_dim=1, equip=False)
    random_variable = UnivariateNormalDistributionsRandomVariable(space)
    testing_data = UnivariateNormalDistributionsTestData(
        space, excluded_methods=("sample_belongs_to_support",)
    )


class TestUnivariateNormalToPoincareHalfSpaceDiffeo(
    DiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    domain_space = NormalDistributions(sample_dim=1, equip=False)
    image_space = PoincareHalfSpace(dim=2, equip=False)
    diffeo = UnivariateNormalToPoincareHalfSpaceDiffeo()

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestUnivariateNormalMetric(
    UnivariateNormalMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = NormalDistributions(sample_dim=1)
    testing_data = UnivariateNormalMetricTestData(
        space,
        excluded_methods=() if autodiff_backend() else ("sectional_curvature_value",),
        generator_kwargs={"amplitude": 5.0},
    )


class TestCenteredNormalDistributions(
    InformationManifoldMixinTestCase,
    VectorSpaceOpenSetTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = NormalDistributions(
        sample_dim=random.randint(3, 5), distribution_type="centered", equip=False
    )
    random_variable = SharedMeanNormalDistributionsRandomVariable(space)
    testing_data = CenteredNormalDistributionsTestData(
        space, excluded_methods=("sample_belongs_to_support",)
    )


@pytest.mark.redundant
class TestCenteredNormalMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = NormalDistributions(
        sample_dim=random.randint(3, 5), distribution_type="centered"
    )
    testing_data = SPDAffineMetricTestData(space)


class TestDiagonalNormalDistributions(
    InformationManifoldMixinTestCase,
    VectorSpaceOpenSetTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = NormalDistributions(
        sample_dim=random.randint(2, 5), distribution_type="diagonal", equip=False
    )
    random_variable = DiagonalNormalDistributionsRandomVariable(space)
    testing_data = DiagonalNormalDistributionsTestData(
        space, excluded_methods=("sample_belongs_to_support",)
    )


class TestDiagonalNormalMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = NormalDistributions(
        sample_dim=random.randint(2, 5), distribution_type="diagonal"
    )
    testing_data = RiemannianMetricTestData(
        space,
        excluded_methods=(
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
        generator_kwargs={"amplitude": 10.0},
    )


class TestGeneralNormalDistributions(
    InformationManifoldMixinTestCase,
    ProductManifoldTestCase,
    metaclass=GeometricDataBasedParametrizer,
):
    space = NormalDistributions(sample_dim=random.randint(3, 5), equip=False)
    random_variable = MultivariateNormalDistributionsRandomVariable(space)
    testing_data = GeneralNormalDistributionsTestData(
        space, excluded_methods=("sample_belongs_to_support",)
    )


@pytest.mark.redundant
class TestGeneralNormalMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = NormalDistributions(sample_dim=random.randint(3, 5))
    testing_data = RiemannianMetricTestData(space)
