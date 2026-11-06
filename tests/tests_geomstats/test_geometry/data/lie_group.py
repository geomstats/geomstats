from .manifold import ManifoldTestData
from .mixins import GroupExpMixinsTestData

_AT_IDENTITY = "random_tangent_vec_at_identity"


def _tangent_translation_map_evaluator(left, inverse):
    def evaluate(op, point, tangent_vec):
        return op(point, left=left, inverse=inverse)(tangent_vec)

    return evaluate


class _LieGroupMixinsTestData(GroupExpMixinsTestData):
    def compose_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("point_a", "point_b"),
        )

    def inverse_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
        )

    def log_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "point"),
        )

    def tangent_translation_map_vec_test_data(self):
        data = []
        for left in [True, False]:
            for inverse in [True, False]:
                data.extend(
                    self.generate_vectorization_data(
                        arg_names=("point", "tangent_vec"),
                        op_evaluator=_tangent_translation_map_evaluator(
                            left, inverse
                        ),
                        left=left,
                        inverse=inverse,
                    )
                )
        return data

    def lie_bracket_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
        )

    def compose_with_inverse_is_identity_test_data(self):
        return self.generate_random_data(arg_names="point")

    def compose_with_identity_is_itself_test_data(self):
        return self.generate_random_data(arg_names="point")

    def exp_after_log_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "point"))

    def log_after_exp_test_data(self):
        return self.generate_random_data(arg_names=("base_point", "tangent_vec"))

    def to_tangent_at_identity_belongs_to_lie_algebra_test_data(self):
        return self.generate_random_data(
            arg_names="tangent_vec",
            samplers={"tangent_vec": _AT_IDENTITY},
        )

    def tangent_translation_map_is_tangent_test_data(self):
        data = []
        for left in [True, False]:
            data.extend(
                self.generate_random_data(
                    arg_names=("base_point", "tangent_vec", "point"),
                    dependencies={"tangent_vec": "base_point"},
                    left=left,
                )
            )
        return data

    def lie_bracket_belongs_test_data(self):
        return self.generate_random_data(
            arg_names=("tangent_vec_a", "tangent_vec_b"),
            samplers={"tangent_vec_a": _AT_IDENTITY, "tangent_vec_b": _AT_IDENTITY},
        )

    def lie_bracket_antisymmetry_test_data(self):
        return self.generate_random_data(
            arg_names=("tangent_vec_a", "tangent_vec_b"),
            samplers={"tangent_vec_a": _AT_IDENTITY, "tangent_vec_b": _AT_IDENTITY},
        )

    def lie_bracket_jacobi_identity_test_data(self):
        return self.generate_random_data(
            arg_names=("tangent_vec_a", "tangent_vec_b", "tangent_vec_c"),
            samplers={
                "tangent_vec_a": _AT_IDENTITY,
                "tangent_vec_b": _AT_IDENTITY,
                "tangent_vec_c": _AT_IDENTITY,
            },
        )

    def lie_bracket_bilinearity_test_data(self):
        return self.generate_random_data(
            arg_names=("tangent_vec_a", "tangent_vec_b", "tangent_vec_c"),
            samplers={
                "tangent_vec_a": _AT_IDENTITY,
                "tangent_vec_b": _AT_IDENTITY,
                "tangent_vec_c": _AT_IDENTITY,
            },
        )


class MatrixLieGroupTestData(_LieGroupMixinsTestData, ManifoldTestData):
    pass


class LieGroupTestData(_LieGroupMixinsTestData, ManifoldTestData):
    def jacobian_translation_vec_test_data(self):
        data = []
        for left in [True, False]:
            data.extend(
                self.generate_vectorization_data(
                    arg_names="point",
                    op_evaluator=lambda op, point, left=left: op(point, left=left),
                    left=left,
                )
            )
        return data

    def exp_from_identity_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="tangent_vec",
            samplers={"tangent_vec": _AT_IDENTITY},
        )

    def log_from_identity_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names="point",
        )

    def exp_from_identity_after_log_from_identity_test_data(self):
        return self.generate_random_data(arg_names="point")

    def log_from_identity_after_exp_from_identity_test_data(self):
        return self.generate_random_data(
            arg_names="tangent_vec",
            samplers={"tangent_vec": _AT_IDENTITY},
        )
