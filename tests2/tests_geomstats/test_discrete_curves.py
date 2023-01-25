import random

import pytest

from geomstats.geometry.discrete_curves import DiscreteCurves
from geomstats.geometry.euclidean import Euclidean
from geomstats.test.geometry.discrete_curves import DiscreteCurvesTestCase
from geomstats.test.parametrizers import DataBasedParametrizer
from tests2.data.discrete_curves_data import DiscreteCurvesTestData


@pytest.fixture(
    scope="class",
    params=[
        (2, random.randint(5, 10)),
        # (3, random.randint(5, 10)),
    ],
)
def spaces(request):
    dim, k_sampling_points = request.param

    ambient_manifold = Euclidean(dim=dim)
    request.cls.space = DiscreteCurves(
        ambient_manifold, k_sampling_points=k_sampling_points
    )


@pytest.mark.usefixtures("spaces")
class TestDiscreteCurves(DiscreteCurvesTestCase, metaclass=DataBasedParametrizer):
    testing_data = DiscreteCurvesTestData()
