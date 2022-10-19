"""Unit tests for the sub-Riemannian metric class."""


import geomstats.backend as gs
import tests.conftest
from tests.conftest import Parametrizer, TestCase
from tests.data.sub_riemannian_metric_data import TestDataSubRiemannianMetric


class TestSubRiemannianMetric(TestCase, metaclass=Parametrizer):

    testing_data = TestDataSubRiemannianMetric()

    def test_inner_coproduct(
        self, metric, cotangent_vec_a, cotangent_vec_b, base_point, expected
    ):
        result = metric.inner_coproduct(cotangent_vec_a, cotangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_hamiltonian(self, metric, cotangent_vec, base_point, expected):
        state = gs.array([base_point, cotangent_vec])
        result = metric.hamiltonian(state)
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_symp_grad(self, metric, test_state, expected):
        result = metric.symp_grad()(test_state)
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_symp_euler(self, metric, test_state, step_size, expected):
        result = metric.symp_euler(step_size)(test_state)
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_iterate(self, metric, test_state, n_steps, step_size, expected):
        step = metric.symp_euler
        result = metric.iterate(step(step_size), n_steps)(test_state)[-10]
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_symp_flow(self, metric, test_state, n_steps, end_time, expected):
        result = metric.symp_flow(end_time, n_steps)(test_state)[-10]
        self.assertAllClose(result, expected)

    @tests.conftest.autograd_tf_and_torch_only
    def test_exp(self, metric, cotangent_vec, base_point, n_steps):
        result = metric.exp(cotangent_vec, base_point, n_steps=n_steps)
        expected = base_point + cotangent_vec
        self.assertAllClose(result, expected)
