"""Unit tests for the preshape space."""

class TestPreShapeSpace(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.dimension = 4
        self.space = Hypersphere(dim=self.dimension)
        self.metric = self.space.metric
        self.n_samples = 10
        
    def test_belongs(self):
        