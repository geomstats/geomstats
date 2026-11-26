import math

import numpy as np

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.test_cases.backend._utils import get_backend_fnc

from ._base import VectorizableCaseData

rand = gs.random.rand


class BackendTestData(VectorizableCaseData):
    def _get_expected_func(self, case):
        func = get_backend_fnc(case["func_name"])
        kwargs = case.get("kwargs") or {}
        return lambda **arrays: func(*arrays.values(), **kwargs)

    def func_vec_test_data(self):
        data = [
            dict(func_name="trace", a=rand(3, 3)),
            dict(func_name="linalg.cholesky", a=SPDMatrices(3).random_point()),
            dict(func_name="linalg.eigvalsh", a=SymmetricMatrices(3).random_point()),
            dict(func_name="linalg.det", a=gs.eye(3)),
            dict(func_name="linalg.det", a=gs.array([[1.0, 2.0], [2.0, 4.0]])),
            dict(func_name="linalg.logm", a=Matrices.to_diagonal(rand(3, 3))),
            dict(func_name="linalg.solve", a=rand(3, 3) + 3 * gs.eye(3), b=rand(3)),
            # a != b: scipy fallback
            dict(
                func_name="linalg.solve_sylvester",
                a=gs.array([[2.0, 1.0], [0.0, 3.0]]),
                b=gs.array([[1.0, 0.0], [1.0, 2.0]]),
                c=rand(2, 2),
            ),
            # hermitian, not symmetric, as in HPD matrices: scipy fallback
            dict(
                func_name="linalg.solve_sylvester",
                a=gs.array([[2.0 + 0j, 1j], [-1j, 2.0 + 0j]]),
                b=gs.array([[2.0 + 0j, 1j], [-1j, 2.0 + 0j]]),
                c=gs.array([[1.0 + 0j, 2j], [-2j, 3.0 + 0j]]),
            ),
            dict(
                func_name="linalg.quadratic_assignment",
                a=gs.array([[0.0, 1.0, 5.0], [1.0, 0.0, 3.0], [5.0, 3.0, 0.0]]),
                b=gs.array([[0.0, 5.0, 3.0], [5.0, 0.0, 1.0], [3.0, 1.0, 0.0]]),
            ),
            dict(func_name="matmul", a=rand(3, 3), b=rand(3, 3)),
            dict(func_name="matmul", a=rand(3, 4), b=rand(4, 3)),
            dict(func_name="outer", a=rand(3), b=rand(3)),
            dict(func_name="outer", a=rand(3), b=rand(4)),
            dict(func_name="matvec", a=rand(4, 3), b=rand(3)),
            dict(func_name="matvec", a=rand(3, 3), b=rand(3)),
            dict(func_name="vecdot", a=rand(3), b=rand(3)),
            dict(func_name="cross", a=rand(3), b=rand(3)),
            dict(func_name="diag_embed", a=rand(3)),
            dict(func_name="diag_embed", a=rand(3), kwargs={"offset": 1}),
            dict(func_name="tril_embed", a=rand(6)),
            dict(func_name="triu_embed", a=rand(3), kwargs={"k": 1}),
        ]

        return self._vectorize(data)

    def func_test_data(self):
        # logarithm of the reflection [[0, 1], [1, 0]]: i * pi times the
        # projector onto its eigenvector of eigenvalue -1
        half_pi_i = 1j * math.pi / 2
        data = [
            dict(
                func_name="array_from_sparse",
                a=[(0,)],
                b=[1.0],
                c=(2,),
                expected=gs.array([1.0, 0.0]),
            ),
            dict(
                func_name="array_from_sparse",
                a=[(0, 0), (0, 1), (1, 2)],
                b=[1.0, 2.0, 3.0],
                c=(2, 3),
                expected=gs.array([[1.0, 2.0, 0], [0, 0, 3.0]]),
            ),
            dict(
                func_name="array_from_sparse",
                a=[(0, 1, 1), (1, 1, 1)],
                b=[1.0, 2.0],
                c=(2, 2, 2),
                expected=gs.array([[[0.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 2.0]]]),
            ),
            dict(
                func_name="gamma",
                a=gs.array([-2.5, -1.5, -0.5, 0.5]),
                expected=math.sqrt(math.pi)
                * gs.array([-8.0 / 15.0, 4.0 / 3.0, -2.0, 1.0]),
            ),
            dict(
                func_name="tril_embed",
                a=gs.array([1.0, 2.0, 3.0]),
                expected=gs.array([[1.0, 0.0], [2.0, 3.0]]),
            ),
            dict(
                func_name="tril_embed",
                a=gs.array([1.0, 2.0, 3.0]),
                kwargs={"k": -1},
                expected=gs.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 3.0, 0.0]]),
            ),
            dict(
                func_name="triu_embed",
                a=gs.array([1.0, 2.0, 3.0]),
                expected=gs.array([[1.0, 2.0], [0.0, 3.0]]),
            ),
            dict(
                func_name="triu_embed",
                a=gs.array([1.0, 2.0, 3.0]),
                kwargs={"k": 1},
                expected=gs.array([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]),
            ),
            dict(
                func_name="linalg.logm",
                a=gs.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]),
                expected=gs.array(
                    [
                        [0.693147180, 0.0, 0.0],
                        [0.0, 1.098612288, 0.0],
                        [0.0, 0.0, 1.38629436],
                    ]
                ),
            ),
            dict(
                func_name="linalg.logm",
                a=gs.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]]),
                expected=gs.array(
                    [[0.0, 0.0, 0.0], [0.0, 1.609437912, 0.0], [0.0, 0.0, 1.79175946]]
                ),
            ),
            dict(
                func_name="linalg.logm",
                a=gs.array([[0.0, 1.0], [1.0, 0.0]]),
                expected=gs.array([[half_pi_i, -half_pi_i], [-half_pi_i, half_pi_i]]),
            ),
            dict(
                func_name="linalg.logm",
                a=gs.array([[[2.0, 0.0], [0.0, 3.0]], [[0.0, 1.0], [1.0, 0.0]]]),
                expected=gs.array(
                    [
                        [[0.693147180 + 0j, 0j], [0j, 1.098612288 + 0j]],
                        [[half_pi_i, -half_pi_i], [-half_pi_i, half_pi_i]],
                    ]
                ),
            ),
            dict(
                func_name="linalg.logm",
                a=gs.array([[2.0 + 0j, 1j], [-1j, 2.0 + 0j]]),
                expected=math.log(3.0)
                / 2
                * gs.array([[1.0 + 0j, 1j], [-1j, 1.0 + 0j]]),
            ),
            dict(
                func_name="linalg.expm",
                a=gs.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]),
                expected=gs.array(
                    [
                        [7.38905609, 0.0, 0.0],
                        [0.0, 20.0855369, 0.0],
                        [0.0, 0.0, 54.5981500],
                    ]
                ),
            ),
            dict(
                func_name="linalg.expm",
                a=gs.array([[0.0, -1.0], [1.0, 0.0]]),
                expected=gs.array(
                    [
                        [math.cos(1.0), -math.sin(1.0)],
                        [math.sin(1.0), math.cos(1.0)],
                    ]
                ),
            ),
            # b relabels the nodes of a by [2, 0, 1]; distinct weights make the
            # optimal alignment unique: the inverse permutation [1, 2, 0]
            dict(
                func_name="linalg.quadratic_assignment",
                a=gs.array([[0.0, 1.0, 5.0], [1.0, 0.0, 3.0], [5.0, 3.0, 0.0]]),
                b=gs.array([[0.0, 5.0, 3.0], [5.0, 0.0, 1.0], [3.0, 1.0, 0.0]]),
                kwargs={"options": {"maximize": True}},
                expected=gs.array([1, 2, 0]),
            ),
            # a = b symmetric positive definite: x = a^{-1} / 2 for q = I
            dict(
                func_name="linalg.solve_sylvester",
                a=gs.array([[2.0, 1.0], [1.0, 2.0]]),
                b=gs.array([[2.0, 1.0], [1.0, 2.0]]),
                c=gs.eye(2),
                expected=gs.array([[2.0, -1.0], [-1.0, 2.0]]) / 6.0,
            ),
            # a = b singular: minimum-norm solution
            dict(
                func_name="linalg.solve_sylvester",
                a=gs.array([[1.0, 0.0], [0.0, 0.0]]),
                b=gs.array([[1.0, 0.0], [0.0, 0.0]]),
                c=gs.array([[0.0, 1.0], [-1.0, 0.0]]),
                expected=gs.array([[0.0, 1.0], [-1.0, 0.0]]),
            ),
            # a != b: (a_ii + b) x_i = q_i
            dict(
                func_name="linalg.solve_sylvester",
                a=gs.array([[1.0, 0.0], [0.0, 2.0]]),
                b=gs.array([[3.0]]),
                c=gs.array([[4.0], [5.0]]),
                expected=gs.array([[1.0], [1.0]]),
            ),
            dict(
                func_name="trace",
                a=gs.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]),
                expected=gs.array([5.0, 13.0]),
            ),
            dict(
                func_name="diag_embed",
                a=gs.array([1.0, 2.0]),
                kwargs={"offset": 1},
                expected=gs.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]),
            ),
            dict(
                func_name="diag_embed",
                a=gs.array([1.0, 2.0]),
                kwargs={"offset": -1},
                expected=gs.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
            ),
            dict(
                func_name="vecdot",
                a=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                b=gs.eye(2),
                kwargs={"axis": 0},
                expected=gs.array([1.0, 4.0]),
            ),
            # without axis, flip reverses all axes
            dict(
                func_name="flip",
                a=gs.array([[1.0, 2.0], [3.0, 4.0]]),
                expected=gs.array([[4.0, 3.0], [2.0, 1.0]]),
            ),
            dict(
                func_name="linalg.matrix_rank",
                a=gs.array([[1.0, 0.0], [0.0, 1e-3]]),
                expected=gs.array(2),
            ),
            dict(
                func_name="linalg.matrix_rank",
                a=gs.array([[1.0, 0.0], [0.0, 1e-3]]),
                kwargs={"tol": 1e-2},
                expected=gs.array(1),
            ),
            # without axis, amin reduces over all axes
            dict(func_name="amin", a=gs.array([[1.0, 2.0], [0.0, 3.0]]), expected=gs.array(0.0)),
            dict(
                func_name="amin",
                a=gs.array([[1.0, 2.0], [0.0, 3.0]]),
                kwargs={"axis": -1},
                expected=gs.array([1.0, 0.0]),
            ),
            # without axis, take indexes the flattened array
            dict(
                func_name="take",
                a=gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                b=gs.array([[0, 5]]),
                expected=gs.array([[1.0, 6.0]]),
            ),
            dict(
                func_name="take",
                a=gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                b=1,
                kwargs={"axis": -1},
                expected=gs.array([2.0, 5.0]),
            ),
            dict(func_name="argmax", a=gs.array([False, True, False]), expected=gs.array(1)),
            dict(
                func_name="clip",
                a=gs.array([0.0, 1.0, 2.0]),
                kwargs={"min": 0.5, "max": 1.5},
                expected=gs.array([0.5, 1.0, 1.5]),
            ),
            # repeated indices are summed
            dict(
                func_name="scatter_add",
                a=gs.zeros(3),
                kwargs={
                    "dim": 0,
                    "index": gs.array([0, 2, 0]),
                    "src": gs.array([1.0, 2.0, 3.0]),
                },
                expected=gs.array([4.0, 0.0, 2.0]),
            ),
            dict(
                func_name="scatter_add",
                a=gs.ones((2, 3)),
                kwargs={
                    "dim": -1,
                    "index": gs.array([[0, 0, 1], [2, 2, 2]]),
                    "src": gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                },
                expected=gs.array([[4.0, 4.0, 1.0], [1.0, 1.0, 16.0]]),
            ),
            dict(
                func_name="expand_dims_to",
                a=gs.array(1.0),
                b=1,
                expected=gs.array([1.0]),
            ),
            dict(
                func_name="expand_dims_to",
                a=gs.array([1.0, 2.0]),
                b=2,
                expected=gs.array([[1.0, 2.0]]),
            ),
            dict(
                func_name="expand_dims_to",
                a=gs.array([1.0, 2.0]),
                b=2,
                kwargs={"axis": 1},
                expected=gs.array([[1.0], [2.0]]),
            ),
            dict(
                func_name="expand_dims_to",
                a=gs.ones((2, 2)),
                b=2,
                expected=gs.ones((2, 2)),
            ),
            dict(
                func_name="take",
                a=gs.array([0, 1]),
                b=gs.array([0, 0, 1]),
                expected=gs.array([0, 0, 1]),
            ),
            dict(
                func_name="take",
                a=gs.array([[0, 1], [2, 3]]),
                b=gs.array([0, 0, 1]),
                kwargs={"axis": 0},
                expected=gs.array([[0, 1]] * 2 + [[2, 3]]),
            ),
            dict(
                func_name="take",
                a=gs.array([[0, 1], [2, 3]]),
                b=gs.array([0, 0, 1]),
                kwargs={"axis": 1},
                expected=gs.matrix_transpose(gs.array([[0, 2]] * 2 + [[1, 3]])),
            ),
            dict(
                func_name="take",
                a=gs.array([[0, 1], [2, 3]]),
                b=0,
                kwargs={"axis": 1},
                expected=gs.array([0, 2]),
            ),
        ]

        return data

    def func_is_reproducible_test_data(self):
        return [
            dict(func_name="random.choice", args=(gs.arange(10.0), 3)),
            dict(func_name="random.normal", args=(0.0, 1.0, 3)),
            dict(func_name="random.uniform", args=(0.0, 1.0, 3)),
        ]

    def func_out_shape_test_data(self):
        return [
            # without size, a random function gives a single sample
            dict(func_name="random.normal", args=(), expected=()),
            dict(func_name="random.uniform", args=(), expected=()),
            dict(
                func_name="random.multivariate_normal",
                args=(gs.zeros(2), gs.eye(2)),
                expected=(2,),
            ),
            dict(func_name="random.choice", args=(gs.ones((4, 3)),), expected=(3,)),
            dict(func_name="random.choice", args=(gs.ones((4, 3)), 2), expected=(2, 3)),
            dict(func_name="random.choice", args=(5,), expected=()),
            # take keeps the axes of size one
            dict(func_name="take", args=(gs.ones((1, 3)), 0, -1), expected=(1,)),
            dict(func_name="take", args=(gs.ones((2, 3)), [[0, 1]], 1), expected=(2, 1, 2)),
        ]

    def func_out_type_test_data(self):
        data = [
            dict(func_name="shape", args=(gs.ones(3),), expected=tuple),
            dict(func_name="to_numpy", args=(gs.ones(3),), expected=np.ndarray),
            dict(
                func_name="to_numpy",
                args=(gs.conj(gs.ones(3, dtype=gs.get_default_cdtype())),),
                expected=np.ndarray,
            ),
        ]

        return data

    def func_out_equal_test_data(self):
        data = [
            dict(func_name="shape", args=(1,), expected=()),
            dict(func_name="shape", args=([1, 2],), expected=(2,)),
            dict(func_name="shape", args=(gs.ones(3),), expected=(3,)),
            dict(func_name="shape", args=(gs.ones((3, 3)),), expected=(3, 3)),
            dict(func_name="take", args=[gs.array([0, 1]), 0], expected=0),
            dict(func_name="is_array", args=[gs.ones(2)], expected=True),
            dict(func_name="is_array", args=([1, 2],), expected=False),
            dict(func_name="is_array", args=(1,), expected=False),
        ]

        return data

    def func_raises_error_test_data(self):
        data = [
            dict(func_name="matmul", a=rand(1), b=rand(1)),
            dict(func_name="matmul", a=rand(2, 3, 3), b=rand(2, 3)),
            dict(func_name="matmul", a=rand(2, 3, 3), b=rand(3, 3, 3)),
            dict(func_name="matvec", a=rand(3, 2), b=rand(3)),
            dict(func_name="vecdot", a=rand(4), b=rand(3)),
            dict(func_name="vecdot", a=rand(3, 4), b=rand(3)),
            dict(func_name="cross", a=rand(4), b=rand(4)),
            dict(func_name="expand_dims_to", a=gs.ones((2, 2)), b=1),
        ]

        return data

    def compose_with_inverse_test_data(self):
        data = [
            dict(
                func_name_1="linalg.logm",
                func_name_2="linalg.expm",
                a=Matrices.to_diagonal(rand(3, 3)),
            ),
            dict(
                func_name_1="linalg.expm",
                func_name_2="linalg.logm",
                a=Matrices.to_diagonal(rand(3, 3)),
            ),
            dict(
                func_name_1="linalg.logm",
                func_name_2="linalg.expm",
                a=SpecialOrthogonal(n=3).random_point(2),
            ),
        ]
        data += [
            dict(
                func_name_1="diag_embed",
                func_name_2="diagonal",
                a=rand(2, 3),
                kwargs={"offset": offset},
            )
            for offset in (0, 1, -2)
        ]
        data += [
            dict(
                func_name_1=f"{tri}_embed",
                func_name_2=f"{tri}_to_vec",
                a=rand(2, 6 if k == 0 else 3),
                kwargs={"k": k},
            )
            for tri, k in (("tril", 0), ("tril", -1), ("triu", 0), ("triu", 1))
        ]
        data += [
            dict(
                func_name_1="tril_to_vec",
                func_name_2="tril_embed",
                a=gs.tril(rand(2, 3, 3)),
            ),
            dict(
                func_name_1="triu_to_vec",
                func_name_2="triu_embed",
                a=gs.triu(rand(2, 3, 3), k=1),
                kwargs={"k": 1},
            ),
        ]
        return data
