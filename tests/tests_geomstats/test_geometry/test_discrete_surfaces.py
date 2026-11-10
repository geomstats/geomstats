import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.discrete_surfaces import DiscreteSurfaces, L2SurfacesMetric
from geomstats.test.test_case import pytorch_backend, torch_only
from geomstats.test_cases.geometry.discrete_surfaces import (
    DiscreteSurfacesTestCase,
    SurfaceTestCase,
)
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.discrete_surfaces import (
    DiscreteCubesTestData,
    ElasticMetricTestData,
    QuotientElasticMetricTestData,
    SurfaceTestData,
)
from .data.riemannian_metric import RiemannianMetricTestData


def _load_cube():
    vertices, faces = data_utils.load_cube()
    return gs.array(vertices, dtype=gs.float64), gs.array(faces)


@pytest.mark.smoke
class TestSurface(SurfaceTestCase, metaclass=DataBasedParametrizer):
    testing_data = SurfaceTestData()


@torch_only
class TestDiscreteCubes(
    DiscreteSurfacesTestCase, metaclass=GeometricDataBasedParametrizer
):
    _, _faces = _load_cube()
    space = DiscreteSurfaces(_faces, equip=False)
    testing_data = DiscreteCubesTestData(space)


@torch_only
class TestElasticMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _vertices, _faces = _load_cube()
    if pytorch_backend():
        space = DiscreteSurfaces(_faces)
        space.metric.log_solver.n_nodes = 100
        space.metric.exp_solver.n_steps = 100
    else:
        space = None

    testing_data = ElasticMetricTestData(
        space,
        point_counts=[1],
        generator_kwargs={"point": _vertices, "amplitude": 10.0},
        method_marks={"slow": ("exp_after_log",), "xfail": ("exp_after_log",)},
        method_defaults={"exp_after_log": {"atol": 1e-1}},
    )


class TestL2SurfacesMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _, _faces = _load_cube()
    space = DiscreteSurfaces(_faces, equip=False).equip_with_metric(L2SurfacesMetric)
    testing_data = RiemannianMetricTestData(space, point_counts=[1])


@torch_only
@pytest.mark.slow
@pytest.mark.smoke
class TestQuotientElasticMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    _vertices, _faces = _load_cube()
    if pytorch_backend():
        _total_space = DiscreteSurfaces(_faces)
        _total_space.equip_with_group_action("reparametrizations")
        _total_space.equip_with_quotient()
        space = _total_space.quotient
    else:
        space = None

    testing_data = QuotientElasticMetricTestData(
        space,
        point_counts=[1],
        generator_kwargs={"point": _vertices, "amplitude": 10.0},
    )

    def test_log_runs(self, point, base_point):
        """Check the logarithm runs.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices, 3]
            Surface.
        base_point : array-like, shape=[..., n_vertices, 3]
            Surface.
        """
        self.space.metric.log(point, base_point)

    def test_geodesic_bvp_runs(self, point, base_point):
        """Check the boundary value geodesic runs.

        Parameters
        ----------
        point : array-like, shape=[..., n_vertices, 3]
            End surface.
        base_point : array-like, shape=[..., n_vertices, 3]
            Initial surface.
        """
        self.space.metric.geodesic(base_point, end_point=point)
