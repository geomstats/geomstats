from abc import ABC


class PointTransformer(ABC):
    def transform_point(self, point):
        raise NotImplementedError("`transform_point` not implemented")

    def transform_tangent_vec(self, tangent_vec, base_point):
        raise NotImplementedError("`transform_tangent_vec` not implemented")

    def inverse_transform_point(self, other_point):
        raise NotImplementedError("`inverse_transform_point` not implemented")

    def inverse_transform_tangent_vec(self, other_tangent_vec, other_base_point):
        raise NotImplementedError("`inverse_transform_tangent_vec` not implemented")


class IdentityPointTransformer(PointTransformer):
    def transform_point(self, point):
        return point

    def transform_tangent_vec(self, tangent_vec, base_point):
        return tangent_vec

    def inverse_transform_point(self, other_point):
        return other_point

    def inverse_transform_tangent_vec(self, other_tangent_vec, other_base_point):
        return other_tangent_vec
