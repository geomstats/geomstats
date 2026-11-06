import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

from geomstats.geometry.hyperbolic import Hyperbolic, HyperbolicDiffeo
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.test_cases.geometry.diffeo import DiffeoTestCase
from geomstats.test_cases.geometry.riemannian_metric import (
    RiemannianMetricComparisonTestCase,
)

from .data.diffeo import DiffeoTestData
from .data.hyperbolic import ExtrinsicToBall3TestData, HalfSpaceToBall2TestData
from .data.riemannian_metric import RiemannianMetricComparisonTestData


def _pulled_back_space(dim, from_, to):
    """Get a hyperbolic space equipped with the metric of another model.

    Parameters
    ----------
    dim : int
        Dimension of the hyperbolic space.
    from_ : str
        Coordinates type of the space.
    to : str
        Coordinates type of the model whose metric is pulled back.

    Returns
    -------
    space : Manifold
        Hyperbolic space in `from_` coordinates equipped with the pullback of
        the metric in `to` coordinates.
    """
    return Hyperbolic(dim, from_, equip=False).equip_with_metric(
        PullbackDiffeoMetric,
        diffeo=HyperbolicDiffeo(from_, to),
        image_space=Hyperbolic(dim, to),
    )


@pytest.mark.smoke
class TestHalfSpaceToBall2(DiffeoTestCase, metaclass=DataBasedParametrizer):
    diffeo = HyperbolicDiffeo("half-space", "ball")
    testing_data = HalfSpaceToBall2TestData()


@pytest.mark.smoke
class TestExtrinsicToBall3(DiffeoTestCase, metaclass=DataBasedParametrizer):
    diffeo = HyperbolicDiffeo("extrinsic", "ball")
    testing_data = ExtrinsicToBall3TestData()


class TestBallToHalfSpace(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _dim = random.randint(2, 5)
    domain_space = Hyperbolic(_dim, "ball", equip=False)
    image_space = Hyperbolic(_dim, "half-space", equip=False)
    diffeo = HyperbolicDiffeo("ball", "half-space")

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestBallToExtrinsic(DiffeoTestCase, metaclass=GeometricDataBasedParametrizer):
    _dim = random.randint(2, 5)
    domain_space = Hyperbolic(_dim, "ball", equip=False)
    image_space = Hyperbolic(_dim, "extrinsic", equip=False)
    diffeo = HyperbolicDiffeo("ball", "extrinsic")

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestExtrinsicToHalfSpace(
    DiffeoTestCase, metaclass=GeometricDataBasedParametrizer
):
    _dim = random.randint(2, 5)
    domain_space = Hyperbolic(_dim, "extrinsic", equip=False)
    image_space = Hyperbolic(_dim, "half-space", equip=False)
    diffeo = HyperbolicDiffeo("extrinsic", "half-space")

    testing_data = DiffeoTestData(
        domain_space=domain_space,
        image_space=image_space,
        map_=diffeo,
    )


class TestExtrinsicBallMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _dim = random.randint(3, 5)
    space = Hyperbolic(_dim, "extrinsic")
    other_space = _pulled_back_space(_dim, "extrinsic", "ball")

    testing_data = RiemannianMetricComparisonTestData(space)


class TestExtrinsicHalfSpaceMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _dim = random.randint(3, 5)
    space = Hyperbolic(_dim, "extrinsic")
    other_space = _pulled_back_space(_dim, "extrinsic", "half-space")

    testing_data = RiemannianMetricComparisonTestData(space)


class TestBallHalfSpaceMetricCmp(
    RiemannianMetricComparisonTestCase, metaclass=GeometricDataBasedParametrizer
):
    _dim = random.randint(3, 5)
    space = Hyperbolic(_dim, "ball")
    other_space = _pulled_back_space(_dim, "ball", "half-space")

    testing_data = RiemannianMetricComparisonTestData(space)
