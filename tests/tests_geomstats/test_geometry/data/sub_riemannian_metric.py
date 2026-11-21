from polpo.testing.data import CaseData

import geomstats.backend as gs


class SubRiemannianMetricCometricTestData(CaseData):
    def inner_coproduct_test_data(self):
        data = [
            dict(
                cotangent_vec_a=gs.array([1.0, 1.0, 1.0]),
                cotangent_vec_b=gs.array([1.0, 10.0, 1.0]),
                base_point=gs.array([2.0, 1.0, 10.0]),
                expected=gs.array(12.0),
            )
        ]
        return data

    def hamiltonian_test_data(self):
        data = [
            dict(
                cotangent_vec=gs.array([1.0, 1.0, 1.0]),
                base_point=gs.array([2.0, 1.0, 10.0]),
                expected=1.5,
            )
        ]
        return data

    def symp_grad_test_data(self):
        data = [
            dict(
                state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                expected=gs.array([[2.0, 3.0, 4.0], [-0.0, -0.0, -0.0]]),
            )
        ]
        return data

    def symp_euler_test_data(self):
        data = [
            dict(
                state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                step_size=0.01,
                expected=gs.array([[1.02, 1.03, 1.04], [2.0, 3.0, 4.0]]),
            )
        ]
        return data

    def iterate_test_data(self):
        data = [
            dict(
                state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                n_steps=20,
                step_size=0.01,
                expected=gs.array([[1.4, 1.6, 1.8], [2.0, 3.0, 4.0]]),
            )
        ]
        return data

    def symp_flow_test_data(self):
        data = [
            dict(
                state=gs.array([[1.0, 1.0, 1.0], [2.0, 3.0, 4.0]]),
                end_time=1.0,
                n_steps=20,
                expected=gs.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]]),
            )
        ]
        return data

    def exp_test_data(self):
        data = [
            dict(
                cotangent_vec=gs.array([1.0, 1.0, 1.0]),
                base_point=gs.array([2.0, 1.0, 10.0]),
                n_steps=20,
                expected=gs.array([3.0, 2.0, 11.0]),
            )
        ]
        return data


class SubRiemannianMetricFrameTestData(CaseData):
    def sr_sharp_test_data(self):
        data = [
            dict(
                cotangent_vec=gs.array([[0.5, 0.5, 0.5], [2.5, 2.5, 2.5]]),
                base_point=gs.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
                expected=gs.array([[0.5, 0.5, 0.0], [1.25, 3.75, 1.25]]),
            )
        ]
        return data

    def geodesic_test_data(self):
        data = [
            dict(
                initial_point=gs.array([0.0, 0.0, 0.0]),
                initial_cotangent_vec=gs.array([2.5, 2.5, 2.5]),
                times=gs.linspace(0.0, 20, 3),
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
        return data
