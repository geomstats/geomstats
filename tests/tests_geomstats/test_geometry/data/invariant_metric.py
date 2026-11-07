from polpo.testing.data import CaseData

import geomstats.backend as gs
from geomstats.geometry.invariant_metric import _InvariantMetricMatrix
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

from .riemannian_metric import RiemannianMetricTestData

_AT_IDENTITY = "random_tangent_vec_at_identity"


class InvariantMetricTestData(RiemannianMetricTestData):
    def invariance_test_data(self):
        return self.generate_random_data(
            arg_names=("point_a", "point_b", "tangent_vec_a", "tangent_vec_b"),
            dependencies={"tangent_vec_a": "point_a", "tangent_vec_b": "point_a"},
        )


class InvariantMetricMatrixTestData(InvariantMetricTestData):
    def structure_constant_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("tangent_vec_a", "tangent_vec_b", "tangent_vec_c"),
            samplers={
                "tangent_vec_a": _AT_IDENTITY,
                "tangent_vec_b": _AT_IDENTITY,
                "tangent_vec_c": _AT_IDENTITY,
            },
            on_metric=True,
        )

    def dual_adjoint_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("tangent_vec_a", "tangent_vec_b"),
            samplers={"tangent_vec_a": _AT_IDENTITY, "tangent_vec_b": _AT_IDENTITY},
            on_metric=True,
        )

    def connection_vec_test_data(self):
        return self.generate_vectorization_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
            on_metric=True,
        )

    def dual_adjoint_structure_constant_test_data(self):
        return self.generate_random_data(
            arg_names=("tangent_vec_a", "tangent_vec_b", "tangent_vec_c"),
            samplers={
                "tangent_vec_a": _AT_IDENTITY,
                "tangent_vec_b": _AT_IDENTITY,
                "tangent_vec_c": _AT_IDENTITY,
            },
        )

    def connection_is_invariant_test_data(self):
        return self.generate_random_data(
            arg_names=("base_point", "tangent_vec_a", "tangent_vec_b"),
            dependencies={},
            samplers={"tangent_vec_a": _AT_IDENTITY, "tangent_vec_b": _AT_IDENTITY},
        )


def _so3_normal_basis():
    space = SpecialOrthogonal(3, equip=False)
    space.equip_with_metric(_InvariantMetricMatrix, left=True)
    return space.metric.normal_basis(space.lie_algebra.basis)


class InvariantMetricMatrixSO3TestData(CaseData):
    def inner_product_test_data(self):
        algebra = SpecialOrthogonal(3, equip=False).lie_algebra
        data = [
            dict(
                tangent_vec_a=algebra.matrix_representation(gs.array([1.0, 0, 2.0])),
                tangent_vec_b=algebra.matrix_representation(gs.array([1.0, 0, 0.5])),
                base_point=None,
                expected=4.0,
            ),
            dict(
                tangent_vec_a=algebra.matrix_representation(
                    gs.array([[1.0, 0, 2.0], [0, 3.0, 5.0]])
                ),
                tangent_vec_b=algebra.matrix_representation(gs.array([1.0, 0, 0.5])),
                base_point=None,
                expected=gs.array([4.0, 5.0]),
            ),
        ]
        return data

    def structure_constant_test_data(self):
        x, y, z = _so3_normal_basis()
        value = 2.0**0.5 / 2.0
        data = [
            dict(tangent_vec_a=x, tangent_vec_b=y, tangent_vec_c=z, expected=value),
            dict(tangent_vec_a=y, tangent_vec_b=x, tangent_vec_c=z, expected=-value),
            dict(tangent_vec_a=y, tangent_vec_b=z, tangent_vec_c=x, expected=value),
            dict(tangent_vec_a=z, tangent_vec_b=y, tangent_vec_c=x, expected=-value),
            dict(tangent_vec_a=z, tangent_vec_b=x, tangent_vec_c=y, expected=value),
            dict(tangent_vec_a=x, tangent_vec_b=z, tangent_vec_c=y, expected=-value),
            dict(tangent_vec_a=x, tangent_vec_b=x, tangent_vec_c=y, expected=0.0),
        ]
        return data

    def connection_test_data(self):
        x, y, z = _so3_normal_basis()
        data = [
            dict(
                tangent_vec_a=x,
                tangent_vec_b=y,
                base_point=gs.eye(3),
                expected=1.0 / 2**0.5 / 2.0 * z,
            ),
        ]
        return data

    def curvature_test_data(self):
        x, y, z = _so3_normal_basis()
        data = [
            dict(
                tangent_vec_a=x,
                tangent_vec_b=y,
                tangent_vec_c=x,
                base_point=gs.eye(3),
                expected=1.0 / 8 * y,
            ),
            dict(
                tangent_vec_a=gs.stack([x, x]),
                tangent_vec_b=gs.stack([y] * 2),
                tangent_vec_c=gs.stack([x, x]),
                base_point=gs.eye(3),
                expected=gs.array([1.0 / 8 * y] * 2),
            ),
            dict(
                tangent_vec_a=y,
                tangent_vec_b=y,
                tangent_vec_c=z,
                base_point=gs.eye(3),
                expected=gs.zeros_like(z),
            ),
        ]
        return data

    def sectional_curvature_test_data(self):
        x, y, z = _so3_normal_basis()
        data = [
            dict(
                tangent_vec_a=x,
                tangent_vec_b=y,
                base_point=gs.eye(3),
                expected=1.0 / 8,
            ),
            dict(
                tangent_vec_a=y,
                tangent_vec_b=y,
                base_point=gs.eye(3),
                expected=0.0,
            ),
            dict(
                tangent_vec_a=gs.stack([y, y]),
                tangent_vec_b=gs.stack([z] * 2),
                base_point=gs.eye(3),
                expected=gs.array([1.0 / 8, 1.0 / 8]),
            ),
        ]
        return data

    def curvature_derivative_at_identity_test_data(self):
        x, y, z = _so3_normal_basis()
        data = [
            dict(
                tangent_vec_a=x,
                tangent_vec_b=y,
                tangent_vec_c=z,
                tangent_vec_d=x,
                expected=gs.zeros_like(x),
            ),
        ]
        return data
