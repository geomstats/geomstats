"""Kendall Pre-Shape space"""

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.hypersphere import _Hypersphere
from geomstats.geometry.matrices import Matrices

TOLERANCE = 1e-6

class PreShapeSpace(EmbeddedManifold):
    """Class for the Kendall pre-shape space. 
    
    The pre-shape space is the sphere of the space of centered k-ad of 
    landmarks in R^m (for the Frobenius norm). It is endowed with the 
    spherical Procrustes metric d(x, y):= arccos(tr(xy^t)).
    
    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
    """
    
    def __init__(self, k_landmarks, m_ambient):
        super(PreShapeSpace, self).__init__(
            dim=m_ambient * (k_landmarks - 1) - 1,
            embedding_manifold=Matrices(m_ambient, k_landmarks))
        self.embedding_metric = self.embedding_manifold.metric
        self.k_landmarks = k_landmarks
        self.m_ambient = m_ambient
        self.metric = ProcrustesMetric(k_landmarks, m_ambient)
        
    def belongs(self, point, tolerance = TOLERANCE):
        """Test if a point belongs to the pre-shape space.

        This tests whether the point is centered and whether the point's 
        Frobenius norm is 1.

        Parameters
        ----------
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.
        tolerance : float
            Tolerance at which to evaluate norm == 1 and mean == 0.
            Optional, default: 1e-6.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the pre-shape space.
        """ 
        frob_norm = self.embedding_metric.norm(point)
        diff = gs.abs(frob_norm - 1)
        mean = gs.mean(point,axis = -1)
        return gs.less_equal(diff, tolerance)*gs.less_equal(mean, tolerance)
        
    def projection(self, point):
        """Project a point on the pre-shape space.
        
        Parameters
        ----------
        point : array-like, shape=[..., m_ambient, k_landmarks]
            Point in Matrices space.
            
        Returns
        -------
        projected_point : array-like, shape=[..., m_ambient, k_landmarks]
            Point projected on the pre-shape space.
        """
        mean = gs.mean(point,axis = -1)
        centered_point = Matrices.transpose(Matrices.transpose(point)-mean)
        frob_norm = self.embedding_metric.norm(centered_point)
        projected_point = gs.einsum('...,...ij->...ij', 1. / frob_norm, 
        centered_point)

        return projected_point
    
    def random_uniform(self, n_samples=1, tol=1e-6):
        """Sample in the pre-shape space from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        tol : float
            Tolerance.
            Optional, default: 1e-6.

        Returns
        -------
        samples : array-like, shape=[..., m_ambient, k_landmarks]
            Points sampled on the pre-shape space.
        """
        samples = _Hypersphere(
            self.m_ambient*self.k_landmarks-1).random_uniform(n_samples,tol)
        return self.projection(samples) 
  
      
class ProcrustesMetric():
    """Procrustes metric on the pre-shape space.

    Parameters
    ----------
    k_landmarks : int
        Number of landmarks
    m_ambient : int
        Number of coordinates of each landmark.
    """

    def __init__(self, k_landmarks, m_ambient):
        super(ProcrustesMetric, self).__init__(
            dim=m_ambient * (k_landmarks - 1) - 1,
            embedding_manifold=Matrices(m_ambient, k_landmarks))
        self.embedding_metric = self.embedding_manifold.metric
        self._space = PreShapeSpace(k_landmarks, m_ambient)