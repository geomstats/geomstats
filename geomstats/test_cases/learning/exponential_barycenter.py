from geomstats.geometry.euclidean import Euclidean


class EuclideanGroup(Euclidean):
    @staticmethod
    def compose(point_a, point_b):
        return point_a + point_b

    def log(self, point, base_point=None):
        if base_point is None:
            base_point = self.identity

        return point - base_point

    def exp(self, tangent_vec, base_point=None):
        if base_point is None:
            return tangent_vec

        return super().exp(tangent_vec, base_point)

    @staticmethod
    def inverse(point):
        return -point
