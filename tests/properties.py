import geomstats.backend as gs


class ManifoldProperties:
    def projection_shape_and_belongs(self, space_args, data, expected, atol):
        space = self.space(*space_args)
        belongs = space.belongs(space.projection(gs.array(data)), atol)
        self.assertAllClose(gs.all(belongs), gs.array(True))
        self.assertAllClose(gs.shape(belongs), expected)

    def to_tangent_shape_and_is_tangent(self, space_args, data, expected):
        space = self.space(*space_args)
        tangent = space.to_tangent(gs.array(data))
        self.assertAllClose(gs.all(space.is_tangent(tangent)), gs.array(True))
        self.assertAllclose(gs.shape(tangent), expected)


class OpenSetProperites(ManifoldProperties):
    def to_tangent_belongs_ambient_space(self, space_args, data, belongs_atol):
        space = self.space(*space_args)
        result = gs.all(space.ambient_space.belongs(gs.array(data), belongs_atol))
        self.asertAllClose(result, gs.array(True))


class LieGroupProperties:
    def exp_log_composition(self, group_args, tangent_vec, base_point, rtol, atol):
        group = self.group(*group_args)
        exp_point = group.exp_from_identity(gs.array(tangent_vec), gs.array(base_point))
        log_vec = group.log_from_idenity(exp_point, gs.array(base_point))
        self.assertAllClose(log_vec, gs.array(tangent_vec), rtol, atol)

    def log_exp_composition(self, group_args, point, base_point, rtol, atol):
        group = self.group(*group_args)
        log_vec = group.log_from_identity(gs.array(point), gs.array(base_point))
        exp_point = group.exp_from_identity(log_vec, gs.array(base_point))
        self.assertAllClose(exp_point, gs.array(point), rtol, atol)


class VectorSpaceProperties:
    def basis_belongs_and_basis_cardinality(self, algebra_args):
        algebra = self.algebra(*algebra_args)
        basis = algebra.belongs(algebra.basis)
        self.assertAllClose(len(basis), algebra.dim)
        self.assertAllClose(gs.all(basis), gs.array(True))


class LieAlgebraProperties(VectorSpaceProperties):
    def basis_representation_matrix_represenation_composition(
        self, algebra_args, mat_rep, atol, rtol
    ):
        algebra = self.algebra(*algebra_args)
        basis_rep = algebra.basis_representation(gs.array(mat_rep))
        result = algebra.matrix_representation(basis_rep)
        self.assertAllClose(result, gs.array(mat_rep))

    def matrix_representation_basis_represenation_composition(
        self, algebra_args, basis_rep, atol, rtol
    ):
        algebra = self.algebra(*algebra_args)
        mat_rep = algebra.matrix_representation(basis_rep)
        result = algebra.basis_representation(mat_rep)
        self.assertAllClose(result, gs.array(basis_rep))
