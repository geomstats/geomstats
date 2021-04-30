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
class BlueTataManifold(Manifold): # skipcq: PYL-D0002
    def belongs(self, point, atol=gs.atol): # skipcq: PY-D0003
        return True


@TataManifoldFactory.register(color='yellow')
class YellowTataManifold(Manifold): # skipcq: PYL-D0002
    def belongs(self, point, atol=gs.atol): # skipcq: PY-D0003
        return True


@TataManifoldFactory.register(dim=3, color='blue')
class MatrixBlueTataManifold(Manifold): # skipcq: PYL-D0002
    def belongs(self, point, atol=gs.atol): # skipcq: PY-D0003
        return True


@TataManifoldFactory.registerMetric(name='TheName')
class FirstTataMetric(Connection): # skipcq: PYL-D0002
    def __init__(self):
        super().__init__(dim=2)

    # skipcq: PYL-W0613, PYL-R0201, PYL-D0002
    def dummy_method(self, my_arg, my_kwarg=None):
        return 1


@TataManifoldFactory.registerMetric()
class SecondTataMetric(Connection): # skipcq: PYL-D0002
    def __init__(self):
        super().__init__(dim=2)

    # skipcq: PY-D0003
    def dummy_method(self, my_arg, my_kwarg=None):
        return f"{my_arg} , {my_kwarg} {self.dim}"


class TestManifold(geomstats.tests.TestCase): # skipcq: PYL-D0002
    def test_creation(self):
        """Test creation of manifold with the factory."""
        manifold_mat_b = TataManifoldFactory.create(color='blue', dim=3)
        manifold_mat_b2 = TataManifoldFactory.create(dim=3, color='blue')
        manifold_b = TataManifoldFactory.create(dim=2, color='blue')
        manifold_y = TataManifoldFactory.create(dim=2, color='yellow')

        self.assertTrue(manifold_mat_b.__class__ == MatrixBlueTataManifold)
        self.assertTrue(manifold_mat_b2.__class__ == MatrixBlueTataManifold)
        self.assertTrue(manifold_b.__class__ == BlueTataManifold)
        self.assertTrue(manifold_y.__class__ == YellowTataManifold)

    def test_bad_creation(self):
        """Test the failure of creation of a manifold."""
        with self.assertRaises(Exception):
            TataManifoldFactory.create(dim=2, color='grey')

    def test_create_with_metrics(self):
        """Test the creation of manifold with metrics."""
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
        """Test the creation of manifold with non existant metric."""
        m_bad_metric = TataManifoldFactory.create(dim=2,
                                                  color='yellow',
                                                  metrics_names='BadName')
        self.assertIsNotNone(m_bad_metric)

    def test_call_method_on_metric(self):
        """Test calling a method on the metrics of a manifold."""
        m = TataManifoldFactory.create(dim=2,
                                       color='yellow',
                                       metrics_names=['TheName',
                                                      'SecondTataMetric'])
        res = m.call_method_on_metrics('dummy_method', 4, my_kwarg='YES')
        self.assertDictEqual(res,
                             {'TheName': 1,
                              'SecondTataMetric': '4 , YES 2'})
