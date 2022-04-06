from geomstats.geometry.manifold import Manifold

class Flag(Manifold):
    def belongs(self, point, atol=gs.atol):
        pass

    def is_tangent(self, vector, base_point, atol=gs.atol):
        pass

    def to_tangent(self, vector, base_point):
        pass

    def random_point(self, n_samples=1, bound=1.0):
        pass