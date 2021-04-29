"""Unit tests for manifolds Factories."""

import sys

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.manifold import AbstractManifoldFactory, Manifold
from geomstats.geometry.connection import Connection
gs.random.seed(2020)


# first let's create the factory
class TataManifoldFactory(AbstractManifoldFactory):
    """Factory for Tata Manifolds."""

    metrics_creators = {} # These are class variables
    manifolds_creators = {}


@TataManifoldFactory.register(color='blue')
class BlueTataManifold(Manifold):
    def belongs(self, point, atol=gs.atol):
        return True


@TataManifoldFactory.register(color='yellow')
class YellowTataManifold(Manifold):
    def belongs(self, point, atol=gs.atol):
        return True


@TataManifoldFactory.register(dim=3, color='blue')
class MatrixBlueTataManifold(Manifold):
    def belongs(self, point, atol=gs.atol):
        return True


@TataManifoldFactory.registerMetric(name='TheName')
class FirstTataMetric(Connection):
    def __init__(self):
        super().__init__(dim=2)

    def dummy_method(self, my_arg, my_kwarg=None):
        return 1


@TataManifoldFactory.registerMetric()
class SecondTataMetric(Connection):
    def __init__(self):
        super().__init__(dim=2)

    def dummy_method(self, my_arg, my_kwarg=None):
        return f"{my_arg} , {my_kwarg}"


class TestManifold(geomstats.tests.TestCase):
    def test_creation(self):
        manifold_mat_b = TataManifoldFactory.create(color='blue', dim=3)
        manifold_mat_b2 = TataManifoldFactory.create(dim=3, color='blue')
        manifold_b = TataManifoldFactory.create(dim=2, color='blue')
        manifold_y = TataManifoldFactory.create(dim=2, color='yellow')

        assert(manifold_mat_b.__class__ == MatrixBlueTataManifold)
        assert(manifold_mat_b2.__class__ == MatrixBlueTataManifold)
        assert(manifold_b.__class__ == BlueTataManifold)
        assert(manifold_y.__class__ == YellowTataManifold)

    def test_bad_creation(self):
        with self.assertRaises(Exception):
            TataManifoldFactory.create(dim=2, color='grey')

    def test_create_with_metrics(self):
        manifold_with_one_metric = TataManifoldFactory.create(dim=2, color='yellow', metrics_names='TheName')
        manifold_with_metrics = TataManifoldFactory.create(dim=2, color='yellow', metrics_names=['TheName', 'SecondTataMetric'])

        assert(len(manifold_with_one_metric.getMetrics()) == 1)
        assert(manifold_with_one_metric.getMetrics()[0].__class__ == FirstTataMetric)

        assert(len(manifold_with_metrics.getMetrics()) == 2)
        assert(manifold_with_metrics.getMetrics()[0].__class__ == FirstTataMetric)
        assert(manifold_with_metrics.getMetrics()[1].__class__ == SecondTataMetric)

        assert(manifold_with_metrics.getMetrics()[0].manifold == manifold_with_metrics)

    def test_with_bad_metric(self):
        manifold_with_bad_metric = TataManifoldFactory.create(dim=2, color='yellow', metrics_names='BadName')
        self.assertIsNotNone(manifold_with_bad_metric)

    def test_call_method_on_metric(self):
        manifold_with_metrics = TataManifoldFactory.create(dim=2, color='yellow', metrics_names=['TheName', 'SecondTataMetric'])
        res = manifold_with_metrics.call_method_on_metrics('dummy_method', 4, my_kwarg='YES')
        self.assertDictEqual(res, {'TheName': 1, 'SecondTataMetric' : '4 , YES'})
