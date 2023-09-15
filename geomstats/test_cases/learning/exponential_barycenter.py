from geomstats.geometry.euclidean import Euclidean


class EuclideanGroup(Euclidean):
    def compose(self, point_a, point_b):
        return point_a + point_b

    def log(self, point, base_point=None):
        if base_point is None:
            base_point = self.identity

        return point - base_point

    def exp(self, tangent_vec, base_point=None):
        if base_point is None:
            return tangent_vec

        return super().exp(tangent_vec, base_point)

    def inverse(self, point):
        return -point
