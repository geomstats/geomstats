"""Unit tests for the preshape space."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.pre_shape import PreShapeSpace

class TestPreShapeSpace(geomstats.tests.TestCase):
    def setUp(self):
        gs.random.seed(1234)

        self.k_landmarks = 4
        self.m_ambient = 3
        self.space = PreShapeSpace(self.k_landmarks, self.m_ambient)
        self.n_samples = 10
        
    def test_random_uniform_and_belongs(self):
        """Test random uniform and belongs.

        Test that the random uniform method samples
        on the pre-shape space.
        """
        n_samples = self.n_samples
        point = self.space.random_uniform(n_samples)
        result = self.space.belongs(point)
        expected = gs.array([True] * n_samples) 
        
        self.assertAllClose(expected, result)
        
    def test_random_uniform(self):
        point = self.space.random_uniform()

        self.assertAllClose(gs.shape(point), (
            self.m_ambient, self.k_landmarks,))
        
    def test_projection_and_belongs(self):
        point = gs.array(
            [[1., 0., 0.,1.], 
             [0., 1.,0.,1.],
             [0.,0.,1.,1.]])
        proj = self.space.projection(point)
        result = self.space.belongs(proj)
        expected = True
        
        self.assertAllClose(expected, result)