import os

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.sasaki_metric import SasakiMetric
from tests.data_generation import TestData


class SasakiMetricTestData(TestData):
    dim = 2
    sas_sphere_metric = SasakiMetric(
        Hypersphere(dim=dim),
        os.cpu_count(),
    )

    # fix elements in TS
    pu0 = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 1.0]])
    pu1 = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    pu2 = gs.array(
        [[-0.27215095, -0.90632948, 0.32326574], [0.13474934, 0.0, 1.39415392]]
    )
    pu3 = gs.array(
        [[0.15891441, -0.95795476, 0.23889097], [0.2547784, 0.0, 0.92187986]]
    )

    def inner_product_test_data(self):
        _sqrt2 = 1.0 / gs.sqrt(2.0)
        base_point = gs.array([[_sqrt2, -_sqrt2, 0], [_sqrt2, _sqrt2, 1]])
        end_point = gs.stack([self.pu0, self.pu1])

        _log = self.sas_sphere_metric.log(end_point, base_point)

        smoke_data = [
            dict(
                metric=self.sas_sphere_metric,
                tangent_vec_a=_log[0],
                tangent_vec_b=_log[1],
                base_point=base_point,
                expected=-0.61685,
            )
        ]
        return self.generate_tests(smoke_data, [])

    def exp_test_data(self):
        tangent_vec = gs.array(
            [[[gs.pi / 2, 0, 0], [0, 0, 0]], [[0, -gs.pi / 2, 0], [0, 0, 0]]]
        )
        expected = gs.array([self.pu1, self.pu0])

        smoke_data = [
            dict(
                metric=self.sas_sphere_metric,
                tangent_vec=tangent_vec,
                base_point=gs.array([self.pu0, self.pu1]),
                expected=expected,
            )
        ]

        return self.generate_tests(smoke_data, [])

    def log_test_data(self):
        expected = gs.array(
            [[[gs.pi / 2, 0, 0], [0, 0, 0]], [[0, -gs.pi / 2, 0], [0, 0, 0]]]
        )

        smoke_data = [
            dict(
                metric=self.sas_sphere_metric,
                point=gs.array([self.pu1, self.pu0]),
                base_point=gs.array([self.pu0, self.pu1]),
                expected=expected,
            )
        ]
        return self.generate_tests(smoke_data, [])

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
        smoke_data = [
            dict(
                metric=self.sas_sphere_metric,
                initial_point=self.pu2,
                end_point=self.pu3,
                expected=expected,
            )
        ]
        return self.generate_tests(smoke_data, [])
