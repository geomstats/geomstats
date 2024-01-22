import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs

class BrownianMotion:

    def __init__(self, space):
        self._check_metric(space)
        self._check_coordinates(space)
        self.space = space
        self.dim = space.dim
        self.metric = space.metric

    def _check_metric(self, space):
        pass

    def _check_coordinates(self, space):
        pass

    def sample_path(self, end_time, n_steps, initial_point):
        """Generate a sample path of Brownian motion."""

        path = gs.zeros((n_steps, self.dim))
        path[0] = initial_point
        step_size = end_time / n_steps
        for i in range(1, n_steps):
            path[i] = self._step(step_size, path[i-1])

        return path
    
    def _step(self, step_size, current_point):

        sigma = gs.linalg.sqrtm(self.metric.cometric_matrix(current_point))
        christoffels = self.metric.christoffels(current_point)
        cometric_matrix = self.metric.cometric_matrix(current_point)
        
        drift = -0.5 * gs.einsum('klm,lm->k', christoffels, cometric_matrix) * step_size  
        diffusion = gs.einsum('ij, j->i', sigma, gs.random.normal(size=(self.dim,)) * gs.sqrt(step_size))
        
        return current_point + drift + diffusion
    

