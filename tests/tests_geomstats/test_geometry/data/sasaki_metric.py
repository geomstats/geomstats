import geomstats.backend as gs
from geomstats.test.data import TestData


class SasakiMetricSphereTestData(TestData):
    pu0 = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 1.0]])
    pu1 = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    pu2 = gs.array(
        [[-0.27215095, -0.90632948, 0.32326574], [0.13474934, 0.0, 1.39415392]]
    )
    pu3 = gs.array(
        [[0.15891441, -0.95795476, 0.23889097], [0.2547784, 0.0, 0.92187986]]
    )

    def inner_product_test_data(self):
        sqrt2 = 1.0 / gs.sqrt(2.0)
        tangent_vec_a = gs.array(
            [[-5.55360367e-01, -5.55360367e-01, 0.0], [0.0, 0.0, 0.0]]
        )
        tangent_vec_b = -tangent_vec_a
        base_point = gs.array([[sqrt2, -sqrt2, 0], [sqrt2, sqrt2, 1]])
        data = [
            dict(
                tangent_vec_a=tangent_vec_a,
                tangent_vec_b=tangent_vec_b,
                base_point=base_point,
                expected=-0.61685,
            )
        ]
        return self.generate_tests(data)

    def exp_test_data(self):
        tangent_vec = gs.array(
            [[[gs.pi / 2, 0, 0], [0, 0, 0]], [[0, -gs.pi / 2, 0], [0, 0, 0]]]
        )
        expected = gs.stack([self.pu1, self.pu0])

        data = [
            dict(
                tangent_vec=tangent_vec,
                base_point=gs.stack([self.pu0, self.pu1]),
                expected=expected,
            )
        ]

        return self.generate_tests(data)

    def log_test_data(self):
        expected = gs.array(
            [[[gs.pi / 2, 0, 0], [0, 0, 0]], [[0, -gs.pi / 2, 0], [0, 0, 0]]]
        )

        data = [
            dict(
                point=gs.stack([self.pu1, self.pu0]),
                base_point=gs.stack([self.pu0, self.pu1]),
                expected=expected,
            )
        ]
        return self.generate_tests(data)

    def geodesic_discrete_test_data(self):
        expected = gs.array(
            [
                [[-0.27215095, -0.90632948, 0.32326574], [0.13474934, 0.0, 1.39415392]],
                [
                    [-0.13415037, -0.9507542, 0.27941033],
                    [0.17027678, -0.03133438, 1.24635354],
                ],
                [
                    [0.01180225, -0.96796946, 0.25079044],
                    [0.21334916, -0.0301587, 1.08786833],
                ],
                [[0.15891441, -0.95795476, 0.23889097], [0.2547784, 0.0, 0.92187986]],
            ]
        )
        data = [
            dict(
                initial_point=self.pu2,
                end_point=self.pu3,
                expected=expected,
                atol=6e-06,
            )
        ]
        return self.generate_tests(data)
