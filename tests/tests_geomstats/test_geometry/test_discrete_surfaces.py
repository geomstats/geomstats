import pytest

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces
from geomstats.test.parametrizers import DataBasedParametrizer
from geomstats.test.test_case import pytorch_backend, torch_only
from geomstats.test_cases.geometry.discrete_surfaces import (
    DiscreteSurfacesTestCase,
    SurfacesLocalRandomDataGenerator,
    SurfaceTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.discrete_surfaces import (
    DiscreteSurfacesSmokeTestData,
    DiscreteSurfacesTestData,
    ElasticMetricTestData,
    SurfaceTestData,
)


@pytest.mark.smoke
class TestSurface(SurfaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = SurfaceTestData()


@torch_only
class TestDiscreteSurfaces(DiscreteSurfacesTestCase, metaclass=DataBasedParametrizer):
    vertices, faces = data_utils.load_cube()
    vertices = gs.array(vertices, dtype=gs.float64)
    faces = gs.array(faces)
    space = DiscreteSurfaces(faces, equip=False)

    testing_data = DiscreteSurfacesTestData()


@torch_only
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


@torch_only
class TestElasticMetric(RiemannianMetricTestCase, metaclass=DataBasedParametrizer):
    _vertices, _faces = data_utils.load_cube()
    _vertices = gs.array(_vertices, dtype=gs.float64)
    _faces = gs.array(_faces)

    if pytorch_backend():
        space = DiscreteSurfaces(_faces)

        space.metric.log_solver.n_nodes = 100
        space.metric.exp_solver.n_steps = 100

        data_generator = SurfacesLocalRandomDataGenerator(
            space, _vertices, amplitude=10.0
        )
    testing_data = ElasticMetricTestData()
