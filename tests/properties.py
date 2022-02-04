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
    def to_tangent_belongs_ambient_space(self, space_args, data):
        space = self.space(*space_args)
        result = gs.all(space.ambient_space.belongs(gs.array(data)))
        self.asertAllClose(result, gs.array(True))
