import random

import pytest

from geomstats.geometry.hyperboloid import Hyperboloid, HyperboloidMetric
from geomstats.geometry.minkowski import Minkowski
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.base import LevelSetTestCase
from geomstats.test_cases.geometry.hyperbolic import HyperbolicMetricTestCase

from .data.hyperboloid import (
    Hyperboloid2TestData,
    Hyperboloid3TestData,
    HyperboloidMetricTestData,
    HyperboloidTestData,
)


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def spaces(request):
    request.cls.space = Hyperboloid(dim=request.param, equip=False)


@pytest.mark.usefixtures("spaces")
class TestHyperboloid(LevelSetTestCase, metaclass=DataBasedParametrizer):
    testing_data = HyperboloidTestData()


@pytest.mark.smoke
class TestHyperboloid2(LevelSetTestCase, metaclass=DataBasedParametrizer):
    space = Hyperboloid(dim=2)
    testing_data = Hyperboloid2TestData()


@pytest.mark.smoke
class TestHyperboloid3(LevelSetTestCase, metaclass=DataBasedParametrizer):
    space = Hyperboloid(dim=3)
    testing_data = Hyperboloid3TestData()


@pytest.fixture(
    scope="class",
    params=[
        2,
        random.randint(3, 5),
    ],
)
def equipped_spaces(request):
    space = request.cls.space = Hyperboloid(dim=request.param, equip=False)
    space.equip_with_metric(HyperboloidMetric)


@pytest.mark.usefixtures("equipped_spaces")
class TestHyperboloidMetric(HyperbolicMetricTestCase, metaclass=DataBasedParametrizer):
    testing_data = HyperboloidMetricTestData()

    @pytest.mark.random
    def test_inner_product_is_minkowski_inner_product(self, n_points, atol):
        minkowki = Minkowski(self.space.dim + 1)

        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        expected = minkowki.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)
