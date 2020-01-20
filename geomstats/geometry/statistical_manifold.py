"""Statistical Manifolds"""

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geometry.geomstats.riemannian_metric import RiemannianMetric


class StatisticalManifold(Manifold):

    def __init__(self, dim, pdf, n_params):
        assert isinstance(dim, int) and dim > 0
        super(StatisticalManifold).__init__(dimension=dim)
        self.pdf = pdf
        self.n_params = n_params

    def sample(self, point, n_samples=1):
        return self.pdf.rvs(*point, loc=0, scale=1, size=n_samples,
                            random_state=None)

    def sample(self, point, n_samples=1):
        return self.pdf.rvs(*point, loc=0, scale=1, size=n_samples, random_state=None)

    def maximum_likelihood_fit(self, observations, *args, **kwds):
        observations = gs.to_ndarray(observations, to_ndim=2)
        result = []
        for sample in observations:
            output = self.pdf.fit(sample, *args, **kwds)
            result.append(output[:self.n_params])
        return gs.stack(result)

    def potential(self):
        raise NotImplementedError(
                'The potential function is not implemented.')


class FisherMetric(RiemannianMetric, StatisticalManifold, LeviCivitaConnection):

    def __init__(self, dim, pdf, n_params):
        super(StatisticalManifold, self).__init__(dim=dim,
                pdf=pdf, n_params=n_params)

    def inner_product_matrix(self, point):
