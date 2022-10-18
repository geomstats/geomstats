import geomstats.backend as gs
from geomstats.geometry.sub_riemannian_metric import SubRiemannianMetric
from tests.data_generation import TestData


class ExampleMetric(SubRiemannianMetric):
    def __init__(self, dim, dist_dim, default_point_type="vector"):
        super().__init__(
            dim=dim, dist_dim=dist_dim, default_point_type=default_point_type
        )

    def cometric_matrix(self, base_point=None):
        return gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class TestDataSubRiemannianMetric(TestData):
    sub_metric = ExampleMetric(dim=3, dist_dim=2)

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
