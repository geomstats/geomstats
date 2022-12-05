import geomstats.backend as gs
from tests.data_generation import TestData


class SubRiemannianMetricCometricTestData(TestData):
    def inner_coproduct_test_data(self):
        smoke_data = [
            dict(
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
                cotangent_vec=gs.array([1.0, 1.0, 1.0]),
                base_point=gs.array([2.0, 1.0, 10.0]),
                expected=1.5,
            )
        ]
        return self.generate_tests(smoke_data)

    def symp_grad_test_data(self):
        smoke_data = [
            dict(
                test_state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                expected=gs.array([[2.0, 3.0, 4.0], [-0.0, -0.0, -0.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def symp_euler_test_data(self):
        smoke_data = [
            dict(
                test_state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                step_size=0.01,
                expected=gs.array([[1.02, 1.03, 1.04], [2.0, 3.0, 4.0]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def iterate_test_data(self):
        smoke_data = [
            dict(
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
                cotangent_vec=gs.array([1.0, 1.0, 1.0]),
                base_point=gs.array([2.0, 1.0, 10.0]),
                n_steps=20,
            )
        ]
        return self.generate_tests(smoke_data)

    def symp_flow_test_data(self):
        smoke_data = [
            dict(
                test_state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                n_steps=20,
                end_time=1.0,
                expected=gs.array([[2.1, 2.65, 3.2], [2.0, 3.0, 4.0]]),
            )
        ]
        return self.generate_tests(smoke_data)


class SubRiemannianMetricFrameTestData(TestData):
    def sr_sharp_test_data(self):
        smoke_data = [
            dict(
                base_point=gs.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                cotangent_vec=gs.array([[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]]),
                expected=gs.array([[0.5, 0.5, 0.0], [1.25, 3.75, 1.25]]),
            )
        ]
        return self.generate_tests(smoke_data)

    def geodesic_test_data(self):
        smoke_data = [
            dict(
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
