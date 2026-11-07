import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.information_geometry.binomial import (
    BinomialDistributions,
    BinomialDistributionsRandomVariable,
    BinomialMetric,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.binomial import (
    BinomialDistributionsTestCase,
)

from ..test_geometry.data.riemannian_metric import RiemannianMetricTestData
from .data.binomial import (
    Binomial5MetricTestData,
    Binomial7MetricTestData,
    Binomial10MetricTestData,
    BinomialDistributionsTestData,
)


class TestBinomialDistributions(
    BinomialDistributionsTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = BinomialDistributions(n_draws=random.randint(2, 5), equip=False)
    random_variable = BinomialDistributionsRandomVariable(space)
    testing_data = BinomialDistributionsTestData(space)


@pytest.mark.smoke
class TestBinomial5Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = BinomialDistributions(n_draws=5, equip=False)
    space.equip_with_metric(BinomialMetric)
    testing_data = Binomial5MetricTestData()


@pytest.mark.smoke
class TestBinomial7Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = BinomialDistributions(n_draws=7, equip=False)
    space.equip_with_metric(BinomialMetric)
    testing_data = Binomial7MetricTestData()


@pytest.mark.smoke
class TestBinomial10Metric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    space = BinomialDistributions(n_draws=10, equip=False)
    space.equip_with_metric(BinomialMetric)
    testing_data = Binomial10MetricTestData()


class TestBinomialMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = BinomialDistributions(n_draws=random.randint(2, 5))
    testing_data = RiemannianMetricTestData(
        space,
        excluded_methods=(
            "injectivity_radius",
            "parallel_transport_ivp",
            "parallel_transport_bvp",
            "parallel_transport_ivp_transported_is_tangent",
            "parallel_transport_bvp_transported_is_tangent",
            "parallel_transport_ivp_norm",
            "parallel_transport_bvp_norm",
        ),
        # TODO: regularize tangent_vec
        method_marks={"xfail": ("log_after_exp",)},
    )
