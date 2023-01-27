"""Test data for the fisher rao metric."""

import geomstats.backend as gs
from geomstats.information_geometry.exponential import (
    ExponentialDistributions,
    ExponentialMetric,
)
from geomstats.information_geometry.binomial import (
    BinomialDistributions,
    BinomialMetric,
)
from geomstats.information_geometry.poisson import PoissonDistributions, PoissonMetric
from geomstats.information_geometry.geometric import (
    GeometricDistributions,
    GeometricMetric,
)
from geomstats.information_geometry.gamma import GammaDistributions, GammaMetric
from geomstats.information_geometry.beta import BetaDistributions, BetaMetric
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric
from geomstats.information_geometry.normal import (
    NormalDistributions,
    UnivariateNormalDistributions,
    UnivariateNormalMetric,
)
from tests.data_generation import _RiemannianMetricTestData


class FisherRaoMetricTestData(_RiemannianMetricTestData):
    information_manifolds = [
        UnivariateNormalDistributions(),
    ]
    supports = [(-10, 10)]
    Metric = FisherRaoMetric
    metric_args_list = list(zip(information_manifolds, supports))

    shape_list = [metric_args[0].shape for metric_args in metric_args_list]
    space_list = [metric_args[0] for metric_args in metric_args_list]
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
                information_manifold=UnivariateNormalDistributions(),
                support=(-10, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                information_manifold=UnivariateNormalDistributions(),
                support=(-10, 10),
                base_point=gs.array([[1.0, 2.0],[2.0, 3.0]]),
            ),
            dict(
                information_manifold=GammaDistributions(),
                support=(0, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                information_manifold=GammaDistributions(),
                support=(0, 10),
                base_point=gs.array([[1.0, 2.0],[2.0, 3.0]]),
            ),
            dict(
                information_manifold=BetaDistributions(),
                support=(0, 1),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                information_manifold=BetaDistributions(),
                support=(0, 1),
                base_point=gs.array([[1.0, 2.0],[2.0, 3.0]]),
            ),
            dict(
                information_manifold=ExponentialDistributions(),
                support=(0, 10),
                base_point=gs.array([1.0]),
            ),
            dict(
                information_manifold=ExponentialDistributions(),
                support=(0, 10),
                base_point=gs.array([[1.0],[0.5]]),
            ),
            dict(
                information_manifold=BinomialDistributions(10),
                support=(0, 10),
                base_point=gs.array([0.5]),
            ),
            dict(
                information_manifold=BinomialDistributions(10),
                support=(0, 10),
                base_point=gs.array([[0.5],[0.8]]),
            ),            
            dict(
                information_manifold=PoissonDistributions(),
                support=(0, 10),
                base_point=gs.array([1.0]),
            ),
            dict(
                information_manifold=PoissonDistributions(),
                support=(0, 10),
                base_point=gs.array([[1.0],[5.0]]),
            ),
            dict(
                information_manifold=GeometricDistributions(),
                support=(1, 10),
                base_point=gs.array([0.5]),
            ),
            dict(
                information_manifold=GeometricDistributions(),
                support=(1, 10),
                base_point=gs.array([[0.5],[0.8]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_matrix_and_its_inverse_test_data(self):
        smoke_data = [
            dict(
                information_manifold=UnivariateNormalDistributions(),
                support=(-10, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                information_manifold=UnivariateNormalDistributions(),
                support=(-10, 10),
                base_point=gs.array([[1.0, 2.0],[2.0, 3.0]]),
            ),
            dict(
                information_manifold=GammaDistributions(),
                support=(0, 10),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                information_manifold=GammaDistributions(),
                support=(0, 10),
                base_point=gs.array([[1.0, 2.0],[2.0, 3.0]]),
            ),
            dict(
                information_manifold=BetaDistributions(),
                support=(0, 1),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                information_manifold=BetaDistributions(),
                support=(0, 1),
                base_point=gs.array([[1.0, 2.0],[2.0, 3.0]]),
            ),
            dict(
                information_manifold=ExponentialDistributions(),
                support=(0, 10),
                base_point=gs.array([1.0]),
            ),
            dict(
                information_manifold=ExponentialDistributions(),
                support=(0, 10),
                base_point=gs.array([[1.0],[0.5]]),
            ),
            dict(
                information_manifold=BinomialDistributions(10),
                support=(0, 10),
                base_point=gs.array([0.5]),
            ),
            dict(
                information_manifold=BinomialDistributions(10),
                support=(0, 10),
                base_point=gs.array([[0.5],[0.8]]),
            ),            
            dict(
                information_manifold=PoissonDistributions(),
                support=(0, 10),
                base_point=gs.array([1.0]),
            ),
            dict(
                information_manifold=PoissonDistributions(),
                support=(0, 10),
                base_point=gs.array([[1.0],[5.0]]),
            ),
            dict(
                information_manifold=GeometricDistributions(),
                support=(1, 10),
                base_point=gs.array([0.5]),
            ),
            dict(
                information_manifold=GeometricDistributions(),
                support=(1, 10),
                base_point=gs.array([[0.5],[0.8]]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def metric_matrix_and_closed_form_metric_matrix_test_data(self):
        smoke_data = [
            dict(
                information_manifold=UnivariateNormalDistributions(),
                support=(-20, 20),
                closed_form_metric=UnivariateNormalMetric(),
                base_point=gs.array([0.1, 0.8]),
            ),
            dict(
                information_manifold=UnivariateNormalDistributions(),
                support=(-20, 20),
                closed_form_metric=UnivariateNormalMetric(),
                base_point=gs.array([[0.1, 0.8],[0.2, 0.5]]),
            ),
            dict(
                information_manifold=ExponentialDistributions(),
                support=(0, 100),
                closed_form_metric=ExponentialMetric(),
                base_point=gs.array([1.0]),
            ),
            dict(
                information_manifold=ExponentialDistributions(),
                support=(0, 100),
                closed_form_metric=ExponentialMetric(),
                base_point=gs.array([[1.0],[0.5]]),
            ),
            dict(
                information_manifold=GammaDistributions(),
                support=(0, 100),
                closed_form_metric=GammaMetric(),
                base_point=gs.array([0.5, 1.0]),
            ),
            dict(
                information_manifold=GammaDistributions(),
                support=(0, 100),
                closed_form_metric=GammaMetric(),
                base_point=gs.array([[1.0, 2.0],[2.0,3.0]]),
            ),
            dict(
                information_manifold=BetaDistributions(),
                support=(0, 1),
                closed_form_metric=BetaMetric(),
                base_point=gs.array([0.5, 1.0]),
            ),
            dict(
                information_manifold=BetaDistributions(),
                support=(0, 1),
                closed_form_metric=BetaMetric(),
                base_point=gs.array([[0.5, 1.0],[2.0, 3.0]]),
            ),
            # dict(
            #     information_manifold=BinomialDistributions(10),
            #     support=(0, 10),
            #     closed_form_metric=BinomialMetric(10),
            #     base_point=gs.array([0.5]),
            # ),
            # dict(
            #     information_manifold=PoissonDistributions(),
            #     support=(0, 100),
            #     closed_form_metric=PoissonMetric(),
            #     base_point=gs.array([1.0]),
            # ),
            # dict(
            #     information_manifold=GeometricDistributions(),
            #     support=(1, 100),
            #     closed_form_metric=GeometricMetric(),
            #     base_point=gs.array([0.5]),
            # ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_and_closed_form_inner_product_test_data(self):
        normal_dists = NormalDistributions(sample_dim=1)
        smoke_data = [
            dict(
                information_manifold=normal_dists,
                support=(-20, 20),
                closed_form_metric=normal_dists.metric,
                tangent_vec_a=gs.array([1.0, 2.0]),
                tangent_vec_b=gs.array([1.0, 2.0]),
                base_point=gs.array([1.0, 2.0]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def inner_product_derivative_and_closed_form_inner_product_derivative_test_data(self):
        smoke_data = [
            dict(
                information_manifold=ExponentialDistributions(),
                support=(0,100),
                closed_form_derivative=lambda p: gs.expand_dims(gs.expand_dims(-2/p**3,axis=-1),axis=-1),
                base_point=gs.array([0.5])
            ),
            dict(
                information_manifold=ExponentialDistributions(),
                support=(0,200),
                closed_form_derivative=lambda p: gs.expand_dims(gs.expand_dims(-2/p**3,axis=-1),axis=-1),
                base_point=gs.array([[0.2],[0.5]])
            ),
            dict(
                information_manifold=UnivariateNormalDistributions(),
                support=(-20, 20),
                closed_form_derivative=lambda p: gs.array([[[0,-2/p[1]**3], [0,0]],
                                                            [[0,0], [0,-4/p[1]**3]]]) if p.ndim == 1 
                                            else gs.array([[[[0,-2/p_[1]**3], [0,0]],
                                                            [[0,0], [0,-4/p_[1]**3]]] for p_ in p]),
                base_point=gs.array([1.0, 2.0]),
            ),
            dict(
                information_manifold=UnivariateNormalDistributions(),
                support=(-20, 20),
                closed_form_derivative=lambda p: gs.array([[[0,-2/p[1]**3], [0,0]],
                                                            [[0,0], [0,-4/p[1]**3]]]) if p.ndim == 1 
                                            else gs.array([[[[0,-2/p_[1]**3], [0,0]],
                                                            [[0,0], [0,-4/p_[1]**3]]] for p_ in p]),
                base_point=gs.array([[1.0, 2.0],[3.0,1.0]]),
            ),
        ]
        return self.generate_tests(smoke_data)
