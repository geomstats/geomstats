"""Unit tests for the sub-Riemannian metric class."""

import warnings

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric


class ExampleMetric(SubRiemannianMetric):
    def __init__(self, dim, dist_dim, default_point_type="vector"):
        super(ExampleMetric, self).__init__(
            dim=dim, dist_dim=dist_dim, default_point_type=default_point_type
        )

    def cometric_matrix(self, base_point=None):
        return gs.array([[1., 0., 0.],
                         [0., 1.,  0.],
                         [0., 0., 1.]])


class TestSubRiemannianMetric(geomstats.tests.TestCase):
    def setup_method(self):
        warnings.simplefilter("ignore", category=UserWarning)

        self.example_metric = ExampleMetric(dim=3, dist_dim=2)

    def test_inner_coproduct(self):
        cotangent_vec_a = gs.array([1., 1., 1.])
        cotangent_vec_b = gs.array([1., 10., 1.])
        base_point = gs.array([2., 1., 10.])

        result = self.example_metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point)
        expected = gs.array(12.)
        self.assertAllClose(result, expected)

    def test_hamiltonian(self):
        cotangent_vec = gs.array([1., 1., 1.])
        base_point = gs.array([2., 1., 10.])
        state = gs.array([base_point, cotangent_vec])

        result = self.example_metric.hamiltonian(state)
        expected = gs.array([1.5])
        self.assertAllClose(result, expected)

    def test_symp_grad(self):
        test_state = gs.array([[1., 1., 1.], [2., 3., 4.]])
        result = self.example_metric.symp_grad(self.example_metric.hamiltonian)(test_state)
        expected = gs.array([[2.,  3.,  4.], [-0., -0., -0.]])
        self.assertAllClose(result, expected)

    def test_symp_euler(self):
        test_state = gs.array([[1., 1., 1.], [2., 3., 4.]])
        step_size = 0.01
        result = self.example_metric.symp_euler(self.example_metric.hamiltonian, step_size)(test_state)
        expected = gs.array([[1.02, 1.03, 1.04], [2., 3., 4.]])
        self.assertAllClose(result, expected)

    def test_iterate(self):
        test_state = gs.array([[1., 1., 1.], [2., 3., 4.]])
        n_steps = 20
        step_size = 0.01

        step = self.example_metric.symp_euler

        result = self.example_metric.iterate(step(self.example_metric.hamiltonian, step_size), n_steps)(test_state)[-10]
        expected = gs.array([[ 1.22,  1.33, 1.44], [ 2.,  3.,  4.]])
        self.assertAllClose(result, expected)

    def test_symp_flow(self):
        test_state = gs.array([[1., 1., 1.], [2., 3., 4.]])
        end_time = 1.0
        n_steps = 20

        step = self.example_metric.symp_euler

        result = self.example_metric.symp_flow(self.example_metric.hamiltonian, end_time, n_steps)(test_state)[-10]
        expected = gs.array([[2.1, 2.65, 3.2], [2., 3., 4.]])
        self.assertAllClose(result, expected)

    def test_exp(self):
        cotangent_vec = gs.array([1., 1., 1.])
        base_point = gs.array([2., 1., 10.])
        N_STEPS = 20

        result = self.example_metric.exp(
            cotangent_vec, base_point, n_steps=N_STEPS)
        expected = base_point + cotangent_vec
        self.assertAllClose(result, expected)