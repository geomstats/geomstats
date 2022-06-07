import geomstats.backend as gs
from geomstats.geometry.hypersphere import HypersphereMetric
from geomstats.geometry.sasaki_metric import SasakiMetric
from tests.data_generation import TestData


class SasakiMetricTestData(TestData):

    dim = 2
    sas_sphere_metric = SasakiMetric(HypersphereMetric(dim=dim), n_s=3)

    # fix elements in TS
    pu0 = gs.array([[0, -1, 0], [1, 0, 1]])
    pu1 = gs.array([[1, 0, 0], [0, 1, 1]])

    def inner_product_test_data(self):
        _sqrt2 = 1 / gs.sqrt(2)
        base_point = gs.array([[_sqrt2, -_sqrt2, 0], [_sqrt2, _sqrt2, 1]])
        _log = self.sas_sphere_metric.log(gs.array([self.pu0, self.pu1]), base_point)

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
        tangent_vec = gs.array([[[gs.pi / 2, 0, 0], [0, 0, 0]],
                                [[0, -gs.pi / 2, 0], [0, 0, 0]]])
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
        expected = gs.array([[[gs.pi / 2, 0, 0], [0, 0, 0]],
                             [[0, -gs.pi / 2, 0], [0, 0, 0]]])

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
        sqrt32 = gs.sqrt(3) / 2
        expected = [
            [[0, -1, 0], [1, 0, 1]],
            [[.5, -sqrt32, 0], [sqrt32, .5, 1]],
            [[sqrt32, -.5, 0], [.5, sqrt32, 1]],
            [[1, 0, 0], [0, 1, 1]]
        ]
        smoke_data = [
            dict(
                metric=self.sas_sphere_metric,
                initial_point=self.pu0,
                end_point=self.pu1,
                expected=expected,
            )
        ]
        return self.generate_tests(smoke_data, [])
