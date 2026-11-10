import random

import pytest
from polpo.testing.parametrizers import (
    DataBasedParametrizer,
    GeometricDataBasedParametrizer,
)

import geomstats.backend as gs
from geomstats.geometry.functions import HilbertSphere
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase

from .data.functions import HilbertSphereSmokeTestData
from .data.manifold import ManifoldTestData
from .data.riemannian_metric import RiemannianMetricTestData


@pytest.mark.smoke
class TestHilbertSphereSmoke(ManifoldTestCase, metaclass=DataBasedParametrizer):
    space = HilbertSphere(gs.linspace(0, 1, num=50))
    testing_data = HilbertSphereSmokeTestData()


class TestHilbertSphere(ManifoldTestCase, metaclass=GeometricDataBasedParametrizer):
    space = HilbertSphere(gs.linspace(0, 1, num=random.randint(5, 10)))
    testing_data = ManifoldTestData(space)


class TestHilbertSphereMetric(
    RiemannianMetricTestCase, metaclass=GeometricDataBasedParametrizer
):
    space = HilbertSphere(gs.linspace(0, 1, num=random.randint(5, 10)))
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
        method_marks={
            "xfail": (
                "dist_triangle_inequality",
                "exp_after_log",
                "exp_belongs",
                "geodesic_boundary_points",
                "geodesic_bvp_belongs",
                "geodesic_bvp_reverse",
                "geodesic_ivp_belongs",
                "log_after_exp",
                "log_is_tangent",
            )
        },
    )
