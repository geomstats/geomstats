from polpo.testing.data import CaseData

import geomstats.backend as gs


class FisherRaoMetricCmpUnivariateNormalTestData(CaseData):
    def metric_matrix_test_data(self):
        smoke_data = [
            dict(base_point=gs.array([0.1, 0.8])),
            dict(base_point=gs.array([[0.1, 0.8], [1.0, 2.0]])),
        ]

        return smoke_data

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                tangent_vec_a=gs.array([1.0, 2.0]),
                tangent_vec_b=gs.array([1.0, 2.0]),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                tangent_vec_a=gs.array([[1.0, 2.0], [0, 2.0]]),
                tangent_vec_b=gs.array([[1.0, 2.0], [0, 2.0]]),
                base_point=gs.array([[1.0, 2.0], [0, 2.0]]),
            ),
        ]

        return smoke_data

    def inner_product_derivative_matrix_test_data(self):
        smoke_data = [
            dict(base_point=gs.array([1.0, 2.0])),
            dict(base_point=gs.array([[1.0, 2.0], [3.0, 1.0]])),
        ]

        return smoke_data


class FisherRaoMetricCmpExponentialTestData(CaseData):
    def metric_matrix_test_data(self):
        smoke_data = [
            dict(
                base_point=gs.array([1.0]),
            ),
            dict(
                base_point=gs.array([[1.0], [0.5]]),
            ),
        ]

        return smoke_data

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                tangent_vec_a=gs.array([0.5]),
                tangent_vec_b=gs.array([0.5]),
                base_point=gs.array([0.5]),
            ),
            dict(
                tangent_vec_a=gs.array([[0.5], [0.8]]),
                tangent_vec_b=gs.array([[0.5], [0.8]]),
                base_point=gs.array([[0.5], [0.8]]),
            ),
        ]
        return smoke_data

    def inner_product_derivative_matrix_test_data(self):
        smoke_data = [
            dict(
                base_point=gs.array([0.5]),
            ),
            dict(
                base_point=gs.array([[0.2], [0.5]]),
            ),
        ]
        return smoke_data


class FisherRaoMetricCmpGammaTestData(CaseData):
    def metric_matrix_test_data(self):
        smoke_data = [
            dict(
                base_point=gs.array([1.0, 4.0]),
            ),
            dict(
                base_point=gs.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
        ]
        return smoke_data

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                tangent_vec_a=gs.array([1.0, 2.0]),
                tangent_vec_b=gs.array([1.0, 2.0]),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                tangent_vec_a=gs.array([[1.0, 2.0], [3.0, 2.0]]),
                tangent_vec_b=gs.array([[1.0, 2.0], [3.0, 2.0]]),
                base_point=gs.array([[1.0, 2.0], [3.0, 2.0]]),
            ),
        ]
        return smoke_data


class FisherRaoMetricCmpBetaTestData(CaseData):
    def metric_matrix_test_data(self):
        smoke_data = [
            dict(
                base_point=gs.array([0.5, 1.0]),
            ),
            dict(
                base_point=gs.array([[0.5, 1.0], [2.0, 3.0]]),
            ),
        ]
        return smoke_data

    def inner_product_test_data(self):
        smoke_data = [
            dict(
                tangent_vec_a=gs.array([1.0, 2.0]),
                tangent_vec_b=gs.array([1.0, 2.0]),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                tangent_vec_a=gs.array([[1.0, 2.0], [3.0, 2.0]]),
                tangent_vec_b=gs.array([[1.0, 2.0], [3.0, 2.0]]),
                base_point=gs.array([[1.0, 2.0], [3.0, 2.0]]),
            ),
        ]
        return smoke_data
