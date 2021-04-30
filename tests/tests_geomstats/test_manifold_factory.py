"""Unit tests for manifolds Factories."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.connection import Connection
from geomstats.geometry.manifold import AbstractManifoldFactory, Manifold
gs.random.seed(2020)


class TataManifoldFactory(AbstractManifoldFactory):
    """Factory for Tata Manifolds."""

    metrics_creators = {}
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
        return f"{my_arg} , {my_kwarg} {self.dim}"


class TestManifold(geomstats.tests.TestCase):
    def test_creation(self):
        manifold_mat_b = TataManifoldFactory.create(color='blue', dim=3)
        manifold_mat_b2 = TataManifoldFactory.create(dim=3, color='blue')
        manifold_b = TataManifoldFactory.create(dim=2, color='blue')
        manifold_y = TataManifoldFactory.create(dim=2, color='yellow')

        self.assertTrue(manifold_mat_b.__class__ == MatrixBlueTataManifold)
        self.assertTrue(manifold_mat_b2.__class__ == MatrixBlueTataManifold)
        self.assertTrue(manifold_b.__class__ == BlueTataManifold)
        self.assertTrue(manifold_y.__class__ == YellowTataManifold)

    def test_bad_creation(self):
        with self.assertRaises(Exception):
            TataManifoldFactory.create(dim=2, color='grey')

    def test_create_with_metrics(self):
        m_one_metric = TataManifoldFactory.create(dim=2,
                                                  color='yellow',
                                                  metrics_names='TheName')
        m_metrics = TataManifoldFactory.create(dim=2,
                                               color='yellow',
                                               metrics_names=['TheName',
                                                              'SecondTataMetric']) # NOQA

        self.assertTrue(len(m_one_metric.getMetrics()) == 1)
        self.assertTrue(m_one_metric.getMetrics()[0].__class__ == FirstTataMetric)
        self.assertTrue(len(m_metrics.getMetrics()) == 2)
        self.assertTrue(m_metrics.getMetrics()[0].__class__ == FirstTataMetric)
        self.assertTrue(m_metrics.getMetrics()[1].__class__ == SecondTataMetric)
        self.assertTrue(m_metrics.getMetrics()[0].manifold == m_metrics)

    def test_with_bad_metric(self):
        m_bad_metric = TataManifoldFactory.create(dim=2,
                                                  color='yellow',
                                                  metrics_names='BadName')
        self.assertIsNotNone(m_bad_metric)

    def test_call_method_on_metric(self):
        m = TataManifoldFactory.create(dim=2,
                                       color='yellow',
                                       metrics_names=['TheName',
                                                      'SecondTataMetric'])
        res = m.call_method_on_metrics('dummy_method', 4, my_kwarg='YES')
        self.assertDictEqual(res,
                             {'TheName': 1,
                              'SecondTataMetric': '4 , YES 2'})
