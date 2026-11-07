from polpo.testing.data import CaseData
from polpo.testing.geometric import GeometricCaseData
from polpo.testing.lazy import LazyValue

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDAffineMetric, SPDEuclideanMetric
from geomstats.metric_geometry.trees import ForestTopology, Split
from geomstats.metric_geometry.wald_space import (
    _AMBIENT_METRIC_TO_SQUARED_DIST_GRAD,
    Wald,
)
from geomstats.test.random import RandomGroveDataGenerator


class MakePartitionsTestData(CaseData):
    def number_of_topologies_test_data(self):
        data = [
            dict(n_labels=2, expected=1),
            dict(n_labels=3, expected=1),
            dict(n_labels=4, expected=3),
        ]
        return data


class SquaredDistAndGradTestData(CaseData):
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

        return data


class Wald2TestData(CaseData):
    def corr_test_data(self):
        tree = Wald(
            ForestTopology(partition=((0,), (1,)), split_sets=((), ())),
            weights=gs.array([]),
        )

        data = [dict(point=tree, expected=gs.eye(2))]
        return data


class Wald3TestData(CaseData):
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
        return data


class WaldGeodesicSolverTestData(GeometricCaseData):
    """Data for geodesic solvers, with points sampled in a single grove."""

    def _get_data_generator(self, data_space=None):
        return LazyValue(
            lambda: RandomGroveDataGenerator(
                self.space, self.space.random_point().topology
            )
        )

    def discrete_geodesic_reverse_test_data(self):
        return self.generate_random_data(arg_names=("initial_point", "end_point"))
