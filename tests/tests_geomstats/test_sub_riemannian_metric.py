"""Unit tests for the sub-Riemannian metric class."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.heisenberg import HeisenbergVectors
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric
from tests.conftest import Parametrizer, TestCase
from tests.data_generation import TestData

heis = HeisenbergVectors()


def heis_frame(point):
    r"""Compute the frame spanning the Heisenberg distribution."""
    translations = heis.jacobian_translation(point)
    if len(translations.shape) == 3:
        return translations[:, :, 0:2]
    return translations[:, 0:2]


heis_sr = SubRiemannianMetric(dim=3, dist_dim=2, frame=heis_frame)


def trivial_cometric_matrix(base_point):
    r"""Compute a trivial cometric."""
    return gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


ExampleMetric = SubRiemannianMetric(
    dim=3, dist_dim=2, cometric_matrix=trivial_cometric_matrix
)


class TestSubRiemannianMetric(TestCase, metaclass=Parametrizer):
    class TestDataSubRiemannianMetric(TestData):
        sub_metric = ExampleMetric
        sub_metric_heis_frame = heis_sr

        def inner_coproduct_test_data(self):
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

        def sr_sharp_test_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric_heis_frame,
                    base_point=gs.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                    cotangent_vec=gs.array([[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]]),
                    expected=gs.array([[0.5, 0.5, 0.0], [1.25, 3.75, 1.25]]),
                )
            ]
            return self.generate_tests(smoke_data)

        def hamiltonian_test_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    cotangent_vec=gs.array([1.0, 1.0, 1.0]),
                    base_point=gs.array([2.0, 1.0, 10.0]),
                    expected=1.5,
                )
            ]
            return self.generate_tests(smoke_data)

        def symp_grad_test_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    test_state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                    expected=gs.array([[2.0, 3.0, 4.0], [-0.0, -0.0, -0.0]]),
                )
            ]
            return self.generate_tests(smoke_data)

        def symp_euler_test_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    test_state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                    step_size=0.01,
                    expected=gs.array([[1.02, 1.03, 1.04], [2.0, 3.0, 4.0]]),
                )
            ]
            return self.generate_tests(smoke_data)

        def iterate_test_data(self):
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

        def exp_test_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric,
                    cotangent_vec=gs.array([1.0, 1.0, 1.0]),
                    base_point=gs.array([2.0, 1.0, 10.0]),
                    n_steps=20,
                )
            ]
            return self.generate_tests(smoke_data)

        def symp_flow_test_data(self):
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

        def geodesic_test_data(self):
            smoke_data = [
                dict(
                    metric=self.sub_metric_heis_frame,
                    test_initial_point=gs.array([0.0, 0.0, 0.0]),
                    test_initial_cotangent_vec=gs.array([2.5, 2.5, 2.5]),
                    test_times=gs.linspace(0.0, 20, 3),
                    n_steps=1000,
                    expected=gs.array(
                        [
                            [0.00000000e00, 0.00000000e00, 0.00000000e00],
                            [4.07436778e-04, -3.14861045e-01, 2.94971630e01],
                            [2.72046277e-01, -1.30946833e00, 1.00021311e02],
                        ]
                    ),
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
        result = metric.symp_grad(hamiltonian=metric.hamiltonian)(test_state)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_symp_euler(self, metric, test_state, step_size, expected):
        result = metric.symp_euler(hamiltonian=metric.hamiltonian, step_size=step_size)(
            test_state
        )
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_iterate(self, metric, test_state, n_steps, step_size, expected):
        step = metric.symp_euler
        result = metric.iterate(
            step(hamiltonian=metric.hamiltonian, step_size=step_size), n_steps
        )(test_state)[-10]
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_symp_flow(self, metric, test_state, n_steps, end_time, expected):
        result = metric.symp_flow(
            hamiltonian=metric.hamiltonian, end_time=end_time, n_steps=n_steps
        )(test_state)[-10]
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_sr_sharp(self, metric, base_point, cotangent_vec, expected):
        result = metric.sr_sharp(base_point, cotangent_vec)
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_exp(self, metric, cotangent_vec, base_point, n_steps):
        result = metric.exp(cotangent_vec, base_point, n_steps=n_steps)
        expected = base_point + cotangent_vec
        self.assertAllClose(result, expected)

    @geomstats.tests.autograd_tf_and_torch_only
    def test_geodesic(
        self,
        metric,
        test_initial_point,
        test_initial_cotangent_vec,
        test_times,
        n_steps,
        expected,
    ):
        result = metric.geodesic(
            initial_point=test_initial_point,
            initial_cotangent_vec=test_initial_cotangent_vec,
            n_steps=n_steps,
        )(test_times)
        self.assertAllClose(result, expected)
