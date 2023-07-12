"""Test data for the fisher rao metric."""

import geomstats.backend as gs
from geomstats.information_geometry.beta import BetaDistributions
from geomstats.information_geometry.binomial import BinomialDistributions
from geomstats.information_geometry.exponential import ExponentialDistributions
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from geomstats.information_geometry.gamma import GammaDistributions
from geomstats.information_geometry.geometric import GeometricDistributions
from geomstats.information_geometry.normal import UnivariateNormalDistributions
from geomstats.information_geometry.poisson import PoissonDistributions
from tests.data_generation import _RiemannianMetricTestData


class FisherRaoMetricTestData(_RiemannianMetricTestData):
    Metric = FisherRaoMetric

    space_list = [
        UnivariateNormalDistributions(),
    ]
    shape_list = [space.shape for space in space_list]
    metric_args_list = [{"support": (-10, 10)} for _ in space_list]

    n_points_list = [1, 2] * 3
    n_tangent_vecs_list = [1, 2] * 3
    n_points_a_list = [1, 2] * 3
    n_points_b_list = [1]
    alpha_list = [1] * 6
    n_rungs_list = [1] * 6
    scheme_list = ["pole"] * 6

    def inner_product_matrix_shape_test_data(self):
        smoke_data = [
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-10, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-10, 10),
                base_point=gs.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
            dict(
                space=GammaDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                space=GammaDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
            dict(
                space=BetaDistributions(equip=False),
                support=(0, 1),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                space=BetaDistributions(equip=False),
                support=(0, 1),
                base_point=gs.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([1.0]),
            ),
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([[1.0], [0.5]]),
            ),
            dict(
                space=BinomialDistributions(10, equip=False),
                support=(0, 10),
                base_point=gs.array([0.5]),
            ),
            dict(
                space=BinomialDistributions(10, equip=False),
                support=(0, 10),
                base_point=gs.array([[0.5], [0.8]]),
            ),
            dict(
                space=PoissonDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([1.0]),
            ),
            dict(
                space=PoissonDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([[1.0], [5.0]]),
            ),
            dict(
                space=GeometricDistributions(equip=False),
                support=(1, 10),
                base_point=gs.array([0.5]),
            ),
            dict(
                space=GeometricDistributions(equip=False),
                support=(1, 10),
                base_point=gs.array([[0.5], [0.8]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def det_of_inner_product_matrix_test_data(self):
        smoke_data = [
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-10, 10),
                base_point=gs.array([0.0, 0.5]),
            ),
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-10, 10),
                base_point=gs.array([[0.0, 0.5], [1.0, 0.5]]),
            ),
            dict(
                space=GammaDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                space=GammaDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
            dict(
                space=BetaDistributions(equip=False),
                support=(0, 1),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                space=BetaDistributions(equip=False),
                support=(0, 1),
                base_point=gs.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([1.0]),
            ),
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([[1.0], [0.5]]),
            ),
            dict(
                space=BinomialDistributions(10, equip=False),
                support=(0, 10),
                base_point=gs.array([0.5]),
            ),
            dict(
                space=BinomialDistributions(10, equip=False),
                support=(0, 10),
                base_point=gs.array([[0.5], [0.8]]),
            ),
            dict(
                space=PoissonDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([1.0]),
            ),
            dict(
                space=PoissonDistributions(equip=False),
                support=(0, 10),
                base_point=gs.array([[1.0], [5.0]]),
            ),
            dict(
                space=GeometricDistributions(equip=False),
                support=(1, 10),
                base_point=gs.array([0.5]),
            ),
            dict(
                space=GeometricDistributions(equip=False),
                support=(1, 10),
                base_point=gs.array([[0.5], [0.8]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def metric_matrix_and_closed_form_metric_matrix_test_data(self):
        smoke_data = [
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-20, 20),
                base_point=gs.array([0.1, 0.8]),
            ),
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-20, 20),
                base_point=gs.array([[0.1, 0.8], [1.0, 2.0]]),
            ),
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 100),
                base_point=gs.array([1.0]),
            ),
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 100),
                base_point=gs.array([[1.0], [0.5]]),
            ),
            dict(
                space=GammaDistributions(equip=False),
                support=(0, 200),
                base_point=gs.array([1.0, 4.0]),
            ),
            dict(
                space=GammaDistributions(equip=False),
                support=(0, 100),
                base_point=gs.array([[1.0, 2.0], [2.0, 3.0]]),
            ),
            dict(
                space=BetaDistributions(equip=False),
                support=(0, 1),
                base_point=gs.array([0.5, 1.0]),
            ),
            dict(
                space=BetaDistributions(equip=False),
                support=(0, 1),
                base_point=gs.array([[0.5, 1.0], [2.0, 3.0]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_and_closed_form_inner_product_test_data(self):
        smoke_data = [
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-20, 20),
                tangent_vec_a=gs.array([1.0, 2.0]),
                tangent_vec_b=gs.array([1.0, 2.0]),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-20, 20),
                tangent_vec_a=gs.array([[1.0, 2.0], [0, 2.0]]),
                tangent_vec_b=gs.array([[1.0, 2.0], [0, 2.0]]),
                base_point=gs.array([[1.0, 2.0], [0, 2.0]]),
            ),
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 100),
                tangent_vec_a=gs.array([0.5]),
                tangent_vec_b=gs.array([0.5]),
                base_point=gs.array([0.5]),
            ),
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 100),
                tangent_vec_a=gs.array([[0.5], [0.8]]),
                tangent_vec_b=gs.array([[0.5], [0.8]]),
                base_point=gs.array([[0.5], [0.8]]),
            ),
            dict(
                space=BetaDistributions(equip=False),
                support=(0, 1),
                tangent_vec_a=gs.array([1.0, 2.0]),
                tangent_vec_b=gs.array([1.0, 2.0]),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                space=BetaDistributions(equip=False),
                support=(0, 1),
                tangent_vec_a=gs.array([[1.0, 2.0], [3.0, 2.0]]),
                tangent_vec_b=gs.array([[1.0, 2.0], [3.0, 2.0]]),
                base_point=gs.array([[1.0, 2.0], [3.0, 2.0]]),
            ),
            dict(
                space=GammaDistributions(equip=False),
                support=(0, 100),
                tangent_vec_a=gs.array([1.0, 2.0]),
                tangent_vec_b=gs.array([1.0, 2.0]),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                space=GammaDistributions(equip=False),
                support=(0, 100),
                tangent_vec_a=gs.array([[1.0, 2.0], [3.0, 2.0]]),
                tangent_vec_b=gs.array([[1.0, 2.0], [3.0, 2.0]]),
                base_point=gs.array([[1.0, 2.0], [3.0, 2.0]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_derivative_and_closed_form_inner_product_derivative_test_data(
        self,
    ):
        smoke_data = [
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 100),
                closed_form_derivative=lambda p: gs.expand_dims(
                    gs.expand_dims(-2 / p**3, axis=-1), axis=-1
                ),
                base_point=gs.array([0.5]),
            ),
            dict(
                space=ExponentialDistributions(equip=False),
                support=(0, 200),
                closed_form_derivative=lambda p: gs.expand_dims(
                    gs.expand_dims(-2 / p**3, axis=-1), axis=-1
                ),
                base_point=gs.array([[0.2], [0.5]]),
            ),
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-20, 20),
                closed_form_derivative=lambda p: gs.array(
                    [[[0, -2 / p[1] ** 3], [0, 0]], [[0, 0], [0, -4 / p[1] ** 3]]]
                )
                if p.ndim == 1
                else gs.array(
                    [
                        [[[0, -2 / p_[1] ** 3], [0, 0]], [[0, 0], [0, -4 / p_[1] ** 3]]]
                        for p_ in p
                    ]
                ),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                space=UnivariateNormalDistributions(equip=False),
                support=(-20, 20),
                closed_form_derivative=lambda p: gs.array(
                    [[[0, -2 / p[1] ** 3], [0, 0]], [[0, 0], [0, -4 / p[1] ** 3]]]
                )
                if p.ndim == 1
                else gs.array(
                    [
                        [[[0, -2 / p_[1] ** 3], [0, 0]], [[0, 0], [0, -4 / p_[1] ** 3]]]
                        for p_ in p
                    ]
                ),
                base_point=gs.array([[1.0, 2.0], [3.0, 1.0]]),
            ),
        ]
        return self.generate_tests(smoke_data)
