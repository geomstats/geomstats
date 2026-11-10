"""Unit tests for the sub-Riemannian metric class."""

import pytest
from polpo.testing.parametrizers import DataBasedParametrizer

import geomstats.backend as gs
from geomstats.geometry.heisenberg import HeisenbergVectors
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric
from geomstats.test.test_case import autodiff_backend
from geomstats.test_cases.geometry.sub_riemannian_metric import (
    SubRiemannianMetricTestCase,
)

from .data.sub_riemannian_metric import (
    SubRiemannianMetricCometricTestData,
    SubRiemannianMetricFrameTestData,
)

heis = HeisenbergVectors(equip=False)


def heis_frame(point):
    """Compute the frame spanning the Heisenberg distribution."""
    translations = heis.jacobian_translation(point)
    return translations[..., 0:2]


def trivial_cometric_matrix(base_point):
    """Compute a trivial cometric."""
    return gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.smoke
class TestSubRiemannianMetricCometric(
    SubRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = HeisenbergVectors(equip=False)
    space.equip_with_metric(
        SubRiemannianMetric, cometric_matrix=trivial_cometric_matrix
    )
    testing_data = SubRiemannianMetricCometricTestData(
        excluded_methods=()
        if autodiff_backend()
        else ("symp_grad", "symp_euler", "iterate", "symp_flow", "exp")
    )


@pytest.mark.smoke
class TestSubRiemannianMetricFrame(
    SubRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = HeisenbergVectors(equip=False)
    space.equip_with_metric(SubRiemannianMetric, frame=heis_frame)
    testing_data = SubRiemannianMetricFrameTestData(
        excluded_methods=() if autodiff_backend() else ("geodesic",)
    )
