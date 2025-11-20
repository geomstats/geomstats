import geomstats.backend as gs
from geomstats.test.test_case import TestCase


class SubRiemannianMetricTestCase(TestCase):
    def test_inner_coproduct(
        self, cotangent_vec_a, cotangent_vec_b, base_point, expected
    ):
        result = self.space.metric.inner_coproduct(
            cotangent_vec_a, cotangent_vec_b, base_point
        )
        self.assertAllClose(result, expected)

    def test_hamiltonian(self, cotangent_vec, base_point, expected):
        state = gs.array([base_point, cotangent_vec])
        result = self.space.metric.hamiltonian(state)
        self.assertAllClose(result, expected)

    def test_exp(self, cotangent_vec, base_point, n_steps):
        result = self.space.metric.exp(cotangent_vec, base_point, n_steps=n_steps)
        expected = base_point + cotangent_vec
        self.assertAllClose(result, expected)

    def test_geodesic(
        self,
        test_initial_point,
        test_initial_cotangent_vec,
        test_times,
        n_steps,
        expected,
    ):
        result = self.space.metric.geodesic(
            initial_point=test_initial_point,
            initial_cotangent_vec=test_initial_cotangent_vec,
            n_steps=n_steps,
        )(test_times)
        self.assertAllClose(result, expected)

    def test_symp_grad(self, test_state, expected):
        result = self.space.metric.symp_grad(hamiltonian=self.space.metric.hamiltonian)(
            test_state
        )
        self.assertAllClose(result, expected)
