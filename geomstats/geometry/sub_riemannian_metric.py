from abc import ABC

import geomstats.backend as gs
import geomstats.geometry as geometry


class SubRiemannianMetric(ABC):
    """Class for Sub-Riemannian metrics.

    Parameters
    ----------
    dim : int
        Dimension of the manifold.
    default_point_type : str, {'vector', 'matrix'}
        Point type.
        Optional, default: 'vector'.
    """

    def __init__(self, dim, signature=None, default_point_type="vector"):
        super(RiemannianMetric, self).__init__(
            dim=dim, default_point_type=default_point_type
        )

    def metric_matrix(self, point):
        raise NotImplementedError(
            "The computation of the metric matrix" " is not implemented."
        )

    @abstractmethod
    def frame(self, point):
        raise NotImplementedError(
            "The computation of the frame" " is not implemented."
        )

    def hamiltonian(self, state):

        position, momentum = state
        
        frame = self.frame(position)

    
        

    



        
