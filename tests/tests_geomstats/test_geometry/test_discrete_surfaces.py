import pytest

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces, ElasticMetric
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test_cases.geometry.discrete_surfaces import (
    DiscreteSurfacesTestCase,
    ElasticMetricTestCase,
)

from .data.discrete_surfaces import (
    DiscreteSurfacesSmokeTestData,
    DiscreteSurfacesTestData,
    ElasticMetricSmokeTestData,
    ElasticMetricTestData,
)


class TestDiscreteSurfaces(DiscreteSurfacesTestCase, metaclass=DataBasedParametrizer):
    vertices, faces = data_utils.load_cube()
    vertices = gs.array(vertices, dtype=gs.float64)
    faces = gs.array(faces)
    space = DiscreteSurfaces(faces, equip=False)

    testing_data = DiscreteSurfacesTestData()


@pytest.mark.smoke
class TestDiscreteSurfacesSmoke(
    DiscreteSurfacesTestCase, metaclass=DataBasedParametrizer
):
    """Discrete curves smoke test.

    We test this on a space whose initializing
    point is a cube, and we test the function on
    a cube with sides of length 2 centered at the origin.
    The cube is meshed with 12 triangles (2 triangles
    per face.)

    The cube is meshed with triangles, so each face should
    have area 2.
    """

    vertices, faces = data_utils.load_cube()
    vertices = gs.array(vertices, dtype=gs.float64)
    faces = gs.array(faces)
    space = DiscreteSurfaces(faces, equip=False)

    testing_data = DiscreteSurfacesSmokeTestData()


@pytest.mark.skip
class TestElasticMetric(ElasticMetricTestCase, metaclass=DataBasedParametrizer):
    vertices, faces = data_utils.load_cube()
    vertices = gs.array(vertices, dtype=gs.float64)
    faces = gs.array(faces)
    space = DiscreteSurfaces(faces)

    testing_data = ElasticMetricTestData()


@pytest.mark.smoke
class TestElasticMetricSmoke(ElasticMetricTestCase, metaclass=DataBasedParametrizer):
    vertices, faces = data_utils.load_cube()
    vertices = gs.array(vertices, dtype=gs.float64)
    faces = gs.array(faces)
    space = DiscreteSurfaces(faces, equip=False)
    space.equip_with_metric(ElasticMetric, a0=1, a1=1, b1=1, c1=1, d1=1, a2=1)

    testing_data = ElasticMetricSmokeTestData()
