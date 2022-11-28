"""Unit tests for the sub-Riemannian metric class."""

import geomstats.backend as gs
import tests.conftest
from geomstats.geometry.heisenberg import HeisenbergVectors
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric
from tests.conftest import Parametrizer, TestCase
from tests.data.sub_riemannian_metric_data import (
    SubRiemannianMetricCometricTestData,
    SubRiemannianMetricFrameTestData,
)

heis = HeisenbergVectors()


def heis_frame(point):
    """Compute the frame spanning the Heisenberg distribution."""
    translations = heis.jacobian_translation(point)
    return translations[..., 0:2]


def trivial_cometric_matrix(base_point):
    """Compute a trivial cometric."""
    return gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class SubRiemannianMetricTestCase(TestCase):
    def test_inner_coproduct(
        self, cotangent_vec_a, cotangent_vec_b, base_point, expected
    ):
        result = self.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_hamiltonian(self, cotangent_vec, base_point, expected):
        state = gs.array([base_point, cotangent_vec])
        result = self.metric.hamiltonian(state)
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_exp(self, cotangent_vec, base_point, n_steps):
        result = self.metric.exp(cotangent_vec, base_point, n_steps=n_steps)
        expected = base_point + cotangent_vec
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_geodesic(
        self,
        test_initial_point,
        test_initial_cotangent_vec,
        test_times,
        n_steps,
        expected,
    ):
        result = self.metric.geodesic(
            initial_point=test_initial_point,
            initial_cotangent_vec=test_initial_cotangent_vec,
            n_steps=n_steps,
        )(test_times)
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_symp_grad(self, test_state, expected):
        result = self.metric.symp_grad(hamiltonian=self.metric.hamiltonian)(test_state)
        self.assertAllClose(result, expected)


class TestSubRiemannianMetricCometric(
    SubRiemannianMetricTestCase, metaclass=Parametrizer
):
    metric = SubRiemannianMetric(dim=3, cometric_matrix=trivial_cometric_matrix)
    testing_data = SubRiemannianMetricCometricTestData()

    skip_test_geodesic = True

    @tests.conftest.autograd_tf_and_torch_only
    def test_symp_euler(self, test_state, step_size, expected):
        # TODO: migrate test to integrators?
        result = self.metric.symp_euler(
            hamiltonian=self.metric.hamiltonian, step_size=step_size
        )(test_state)
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_iterate(self, test_state, n_steps, step_size, expected):
        # TODO: better way to test this?
        step = self.metric.symp_euler
        result = self.metric.iterate(
            step(hamiltonian=self.metric.hamiltonian, step_size=step_size), n_steps
        )(test_state)[-10]
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_symp_flow(self, test_state, n_steps, end_time, expected):
        # TODO: need this test?
        result = self.metric.symp_flow(
            hamiltonian=self.metric.hamiltonian, end_time=end_time, n_steps=n_steps
        )(test_state)[-10]
        self.assertAllClose(result, expected)


class TestSubRiemannianMetricFrame(SubRiemannianMetricTestCase, metaclass=Parametrizer):
    testing_data = SubRiemannianMetricFrameTestData()
    metric = SubRiemannianMetric(dim=3, frame=heis_frame)

    skip_test_exp = True
    skip_test_hamiltonian = True
    skip_test_inner_coproduct = True
    skip_test_symp_grad = True

    @tests.conftest.autograd_tf_and_torch_only
    def test_sr_sharp(self, base_point, cotangent_vec, expected):
        result = self.metric.sr_sharp(base_point, cotangent_vec)
        self.assertAllClose(result, expected)
