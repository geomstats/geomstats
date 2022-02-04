import geomstats.backend as gs


class ManifoldTestProperties:
    def test_projection_shape_and_belongs(self, space_args, data, expected, atol):
        space = self.space(*space_args)
        belongs = space.belongs(space.projection(gs.array(data)), atol)
        self.assertAllClose(gs.all(belongs), gs.array(True))
        self.assertAllClose(gs.shape(belongs), expected)

    def test_to_tangent_shape_and_is_tangent(self, space_args, data, expected):
        space = self.space(*space_args)
        tangent = space.to_tangent(gs.array(data))
        self.assertAllClose(gs.all(space.is_tangent(tangent)), gs.array(True))
        self.assertAllclose(gs.shape(tangent), expected)


class OpenSetProperites:
    def to_tangent_belongs_ambient_space(self, space_args, data, atol):
        space = self.space(*space_args)
        result = gs.all(space.ambient_space.belongs(gs.array(data), atol))
        self.asertAllClose(result, gs.array(True))


class LieGroupProperties:
    def exp_log_composition_identity(self, group_args, tangent_vec, base_point):
        group = self.group(*group_args)
        exp_point = group.exp_from_identity(gs.array(tangent_vec), gs.array(base_point))
        log_vec = group.log_from_idenity(exp_point)
        self.assertAllClose(log_vec, gs.array(tangent_vec))

    def log_exp_composition_identity(self, group_args, point, base_point):
        group = self.group(*group_args)
        log_vec = group.log_from_identity(gs.array(point), gs.array(base_point))
        exp_point = group.exp_from_identity(log_vec)
        self.assertAllClose(exp_point, gs.array(point))
