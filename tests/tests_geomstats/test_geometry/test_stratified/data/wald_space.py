import pytest

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDAffineMetric, SPDEuclideanMetric
from geomstats.geometry.stratified.trees import ForestTopology, Split
from geomstats.geometry.stratified.wald_space import (
    _AMBIENT_METRIC_TO_SQUARED_DIST_GRAD,
    Wald,
)
from geomstats.test.data import TestData

from .point_set import PointMetricTestData


class MakePartitionsTestData(TestData):
    def number_of_topologies_test_data(self):
        data = [
            dict(n_labels=2, expected=1),
            dict(n_labels=3, expected=1),
            dict(n_labels=4, expected=3),
        ]
        return self.generate_tests(data, marks=(pytest.mark.smoke,))


class SquaredDistAndGradTestData(TestData):
    def value_and_grad_against_autodiff_test_data(self):
        metrics = [SPDAffineMetric, SPDEuclideanMetric]

        data = []
        for Metric in metrics:
            _squared_dist_and_grad = _AMBIENT_METRIC_TO_SQUARED_DIST_GRAD[
                Metric.__name__
            ]
            data.append(
                dict(
                    squared_dist_and_grad=_squared_dist_and_grad,
                    AmbientMetric=Metric,
                ),
            )

        return self.generate_tests(data)


class Wald2TestData(TestData):
    def corr_test_data(self):
        tree = Wald(
            ForestTopology(partition=((0,), (1,)), split_sets=((), ())),
            weights=gs.array([]),
        )

        data = [dict(point=tree, expected=gs.eye(2))]
        return self.generate_tests(data)


class Wald3TestData(TestData):
    def corr_test_data(self):
        partition = ((0, 1, 2),)
        split_sets = ((((0, 1), (2,)), ((0, 2), (1,)), ((0,), (1, 2))),)
        split_sets = [[Split(a, b) for a, b in splits] for splits in split_sets]
        topology = ForestTopology(partition=partition, split_sets=split_sets)
        weights = gs.array([0.1, 0.2, 0.3])
        tree = Wald(topology, weights)

        expected_corr = gs.array(
            [[1.0, 0.56, 0.63], [0.56, 1.0, 0.72], [0.63, 0.72, 1.0]]
        )
        data = [dict(point=tree, expected=expected_corr)]
        return self.generate_tests(data)


class WaldSpaceMetricTestData(PointMetricTestData):
    tolerances = {
        "dist_point_to_itself_is_zero": {"atol": 1e-3},
        "geodesic_bvp_reverse": {"atol": 1e-4},
    }


class WaldGeodesicSolverTestData(TestData):
    N_RANDOM_POINTS = [1]

    tolerances = {
        "discrete_geodesic_reverse": {"atol": 1e-4},
    }

    def discrete_geodesic_reverse_test_data(self):
        return self.generate_random_data()
