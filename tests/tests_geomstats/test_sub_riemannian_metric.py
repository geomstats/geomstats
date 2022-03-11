"""Unit tests for the sub-Riemannian metric class."""


import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric
from tests.conftest import TestCase
from tests.data_generation import TestData
from tests.parametrizers import Parametrizer


class ExampleMetric(SubRiemannianMetric):
    def __init__(self, dim, dist_dim, default_point_type="vector"):
        super(ExampleMetric, self).__init__(
            dim=dim, dist_dim=dist_dim, default_point_type=default_point_type
        )

    def cometric_matrix(self, base_point=None):
        return gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class TestSubRiemannianMetric(TestCase, metaclass=Parametrizer):
    class TestDataSubRiemannianMetric(TestData):
        sub_metric = ExampleMetric(dim=3, dist_dim=2)

        def inner_coproduct_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    cotangent_vec_a=gs.array([1.0, 1.0, 1.0]),
                    cotangent_vec_b=gs.array([1.0, 10.0, 1.0]),
                    base_point=gs.array([2.0, 1.0, 10.0]),
                    expected=gs.array(12.0),
                )
            ]
            return self.generate_tests(smoke_data)

        def hamiltonian_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    cotangent_vec=gs.array([1.0, 1.0, 1.0]),
                    base_point=gs.array([2.0, 1.0, 10.0]),
                    expected=1.5,
                )
            ]
            return self.generate_tests(smoke_data)

        def symp_grad_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    test_state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                    expected=gs.array([[2.0, 3.0, 4.0], [-0.0, -0.0, -0.0]]),
                )
            ]
            return self.generate_tests(smoke_data)

        def symp_euler_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    test_state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                    step_size=0.01,
                    expected=gs.array([[1.02, 1.03, 1.04], [2.0, 3.0, 4.0]]),
                )
            ]
            return self.generate_tests(smoke_data)

        def iterate_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    test_state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                    n_steps=20,
                    step_size=0.01,
                    expected=gs.array([[1.22, 1.33, 1.44], [2.0, 3.0, 4.0]]),
                )
            ]
            return self.generate_tests(smoke_data)

        def exp_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    cotangent_vec=gs.array([1.0, 1.0, 1.0]),
                    base_point=gs.array([2.0, 1.0, 10.0]),
                    n_steps=20,
                )
            ]
            return self.generate_tests(smoke_data)

        def symp_flow_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    test_state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                    n_steps=20,
                    end_time=1.0,
                    expected=gs.array([[2.1, 2.65, 3.2], [2.0, 3.0, 4.0]]),
                )
            ]
            return self.generate_tests(smoke_data)

    testing_data = TestDataSubRiemannianMetric()

    def test_inner_coproduct(
        self, metric, cotangent_vec_a, cotangent_vec_b, base_point, expected
    ):
        result = metric.inner_coproduct(cotangent_vec_a, cotangent_vec_b, base_point)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_hamiltonian(self, metric, cotangent_vec, base_point, expected):
        state = gs.array([base_point, cotangent_vec])
        result = metric.hamiltonian(state)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_symp_grad(self, metric, test_state, expected):
        result = metric.symp_grad()(test_state)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_symp_euler(self, metric, test_state, step_size, expected):
        result = metric.symp_euler(step_size)(test_state)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_iterate(self, metric, test_state, n_steps, step_size, expected):
        step = metric.symp_euler
        result = metric.iterate(step(step_size), n_steps)(test_state)[-10]
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_symp_flow(self, metric, test_state, n_steps, end_time, expected):
        result = metric.symp_flow(end_time, n_steps)(test_state)[-10]
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_exp(self, metric, cotangent_vec, base_point, n_steps):
        result = metric.exp(cotangent_vec, base_point, n_steps=n_steps)
        expected = base_point + cotangent_vec
        self.assertAllClose(result, expected)
