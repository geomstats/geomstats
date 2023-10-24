"""Unit tests for the sub-Riemannian metric class."""

import geomstats.backend as gs
from geomstats.geometry.heisenberg import HeisenbergVectors
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric
from geomstats.test.parametrizers import DataBasedParametrizer
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


class TestSubRiemannianMetricCometric(
    SubRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = HeisenbergVectors(equip=False)
    space.equip_with_metric(
        SubRiemannianMetric, cometric_matrix=trivial_cometric_matrix
    )
    testing_data = SubRiemannianMetricCometricTestData()

    def test_symp_euler(self, test_state, step_size, expected):
        # TODO: migrate test to integrators?
        result = self.space.metric.symp_euler(
            hamiltonian=self.space.metric.hamiltonian, step_size=step_size
        )(test_state)
        self.assertAllClose(result, expected)

    def test_iterate(self, test_state, n_steps, step_size, expected):
        # TODO: better way to test this?
        step = self.space.metric.symp_euler
        result = self.space.metric.iterate(
            step(hamiltonian=self.space.metric.hamiltonian, step_size=step_size),
            n_steps,
        )(test_state)[-10]
        self.assertAllClose(result, expected)

    def test_symp_flow(self, test_state, n_steps, end_time, expected):
        # TODO: need this test?
        result = self.space.metric.symp_flow(
            hamiltonian=self.space.metric.hamiltonian,
            end_time=end_time,
            n_steps=n_steps,
        )(test_state)[-10]
        self.assertAllClose(result, expected)


class TestSubRiemannianMetricFrame(
    SubRiemannianMetricTestCase, metaclass=DataBasedParametrizer
):
    space = HeisenbergVectors(equip=False)
    space.equip_with_metric(SubRiemannianMetric, frame=heis_frame)

    testing_data = SubRiemannianMetricFrameTestData()

    def test_sr_sharp(self, base_point, cotangent_vec, expected):
        result = self.space.metric.sr_sharp(base_point, cotangent_vec)
        self.assertAllClose(result, expected)
