from polpo.testing.data import CaseData

import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.symmetric_matrices import SymmetricMatrices

rand = gs.random.rand

_DTYPES = ["float32", "float64"] if gs.__name__.endswith("pytorch") else ["float64"]
_COMPLEX_DTYPES = {"float32": "complex64", "float64": "complex128"}


def _with_dtypes(cases, key="dtype"):
    """Repeat each case for each dtype.

    Parameters
    ----------
    cases : list[dict]
        Cases.
    key : str
        Name of the dtype argument.

    Returns
    -------
    data : list[dict]
        Cases, one for each dtype.
    """
    return [{**case, key: dtype} for dtype in _DTYPES for case in cases]


def _spd():
    """Create a symmetric positive definite matrix with the default dtype."""
    return (SPDMatrices(2).random_point(),)


def _sym():
    """Create a symmetric matrix with the default dtype."""
    return (SymmetricMatrices(2).random_point(),)


def _diag():
    """Create a diagonal matrix with the default dtype."""
    return (Matrices.to_diagonal(rand(2, 2)),)


def _nonsym():
    """Create a non-symmetric matrix with the default dtype."""
    return (gs.array([[2.0, 1.0], [0.0, 3.0]]),)


def _sylvester():
    """Create the coefficients of a Sylvester equation with the default dtype."""
    a = SPDMatrices(3).random_point()
    return a, a, rand(3, 3)


def _creation_cases():
    """Get cases of array creation functions accepting a dtype."""
    return [
        dict(func_name="array", args=([1.0, 2.0],)),
        # TODO: add int test to arange
        dict(func_name="arange", args=(0.0, 2.0)),
        dict(func_name="linspace", args=(0.0, 1.0)),
        dict(func_name="random.normal", kwargs={"size": 2}),
        dict(
            func_name="random.multivariate_normal",
            kwargs={
                "mean": gs.ones(2),
                "cov": SPDMatrices(2).random_point(),
                "size": 2,
            },
        ),
        dict(func_name="random.uniform", kwargs={"size": 2}),
        dict(func_name="eye", args=(2,)),
        dict(func_name="ones", args=(2,)),
        dict(func_name="zeros", args=(2,)),
        dict(func_name="empty", args=(2,)),
        dict(func_name="random.rand", args=(3,)),
    ]


def _with_dtype_from_array_cases():
    """Get cases of functions of an array accepting a dtype."""
    axis_kwargs = {"axis": 0}
    return [
        dict(func_name="empty_like", shapes=(2,)),
        dict(func_name="ones_like", shapes=(2,)),
        dict(func_name="zeros_like", shapes=(2,)),
        dict(func_name="cumprod", shapes=(2,)),
        dict(func_name="cumsum", shapes=(2,)),
        dict(func_name="mean", shapes=(2,)),
        dict(func_name="mean", shapes=((2, 2),), kwargs=axis_kwargs),
        dict(func_name="sum", shapes=(2,)),
        dict(func_name="sum", shapes=((2, 2),), kwargs=axis_kwargs),
        dict(func_name="std", shapes=(2,)),
        dict(func_name="std", shapes=((2, 2),), kwargs=axis_kwargs),
    ]


class DtypeTestData(CaseData):
    def func_out_dtype_test_data(self):
        data = [
            # TODO: add additional test for int
            dict(
                func_name="where",
                args=([True, False], 20.0, 20.0),
                kwargs={},
                expected=gs.float64,
            ),
            dict(
                func_name="where",
                args=(
                    gs.array([True, False]),
                    gs.ones(2),
                    gs.ones(2, dtype=gs.get_default_cdtype()),
                ),
                kwargs={},
                expected=gs.complex128,
            ),
            dict(
                func_name="where",
                args=(gs.array([True, False]), gs.arange(2), 0.5),
                kwargs={},
                expected=gs.float64,
            ),
        ]
        return data

    def func_out_dtype_with_default_dtype_test_data(self):
        cases = [
            dict(
                func_name="array_from_sparse",
                args=([(0, 0)], [1.0], (2, 2)),
            ),
            *_creation_cases(),
            *[
                dict(func_name=func_name, args=(x,))
                for func_name, x in [
                    ("abs", 1.0),
                    ("angle", 1.0),
                    ("arccos", 0.1),
                    ("arccosh", 2.5),
                    ("arcsin", 0.1),
                    ("arctanh", 0.1),
                    ("ceil", 1.0),
                    ("cos", 1.0),
                    ("cosh", 1.0),
                    ("exp", 1.0),
                    ("floor", 1.0),
                    ("imag", 1.0 + 0.0j),
                    ("log", 1.0),
                    ("real", 1.0),
                    ("sign", 1.0),
                    ("sin", 1.0),
                    ("sinh", 1.0),
                    ("sqrt", 1.0),
                    ("tan", 0.5),
                    ("tanh", 0.5),
                ]
            ],
            dict(func_name="arctan2", args=(1.0, 0.5)),
            dict(func_name="mod", args=(1.0, 2.0)),
            dict(func_name="power", args=(1.0, 1.0)),
            dict(func_name="linalg.cholesky", args=_spd),
            dict(func_name="linalg.eigvalsh", args=_sym),
            dict(func_name="linalg.expm", args=_diag),
            dict(func_name="linalg.expm", args=_spd),
            dict(func_name="linalg.logm", args=_diag),
            dict(func_name="linalg.logm", args=_spd),
            dict(func_name="linalg.logm", args=_nonsym),
            dict(func_name="linalg.sqrtm", args=_spd),
            dict(func_name="linalg.solve_sylvester", args=_sylvester),
        ]
        data = _with_dtypes(cases, key="default_dtype")
        data += [
            dict(
                func_name="array",
                args=([1.0 + 3.0j, 2.0j],),
                default_dtype=dtype,
                expected=_COMPLEX_DTYPES[dtype],
            )
            for dtype in _DTYPES
        ]
        return data

    def func_out_dtype_with_dtype_test_data(self):
        cases = [
            *_creation_cases(),
            *[
                {
                    "func_name": case["func_name"],
                    "args": tuple(gs.ones(shape) for shape in case["shapes"]),
                    **({"kwargs": case["kwargs"]} if "kwargs" in case else {}),
                }
                for case in _with_dtype_from_array_cases()
            ],
        ]
        return _with_dtypes(cases)

    def func_out_dtype_with_input_dtype_test_data(self):
        cases = _with_dtype_from_array_cases() + [
            dict(func_name="copy", shapes=((2, 2),)),
            dict(func_name="diagonal", shapes=((2, 2),)),
            dict(func_name="erf", shapes=(2,)),
            dict(func_name="flatten", shapes=((2, 2),)),
            dict(func_name="trace", shapes=((2, 2),)),
            dict(func_name="tril", shapes=((2, 2),)),
            dict(func_name="tril", shapes=((2, 2),), kwargs={"k": -1}),
            dict(func_name="triu", shapes=((2, 2),)),
            dict(func_name="triu", shapes=((2, 2),), kwargs={"k": 1}),
            dict(func_name="tril_to_vec", shapes=((2, 2),)),
            dict(func_name="triu_to_vec", shapes=((2, 2),)),
            dict(func_name="diag_embed", shapes=(3,)),
            dict(func_name="tril_embed", shapes=(3,)),
            dict(func_name="triu_embed", shapes=(3,)),
            dict(func_name="linalg.vector_norm", shapes=(3,)),
            dict(func_name="linalg.matrix_norm", shapes=((2, 2),)),
            dict(func_name="linalg.qr", shapes=((3, 3),)),
            dict(func_name="cross", shapes=(3, 3)),
            dict(func_name="divide", shapes=(3, 3)),
            dict(
                func_name="divide",
                shapes=(3, 3),
                create_inputs=(gs.ones, gs.zeros),
                kwargs={"ignore_div_zero": True},
            ),
            dict(func_name="vecdot", shapes=(3, 3)),
            dict(func_name="vecdot", shapes=(3, (2, 3))),
            dict(func_name="vecdot", shapes=((2, 3), (2, 3))),
            dict(func_name="matmul", shapes=((2, 2), (2, 2))),
            dict(func_name="matmul", shapes=((2, 3), (3, 2))),
            dict(func_name="matvec", shapes=((3, 3), 3)),
            dict(func_name="matvec", shapes=((3, 3), (2, 3))),
            dict(func_name="matvec", shapes=((2, 3, 3), (2, 3))),
            dict(func_name="outer", shapes=(3, 3)),
            dict(func_name="outer", shapes=(3, (2, 3))),
            dict(func_name="outer", shapes=((2, 3), (2, 3))),
            dict(func_name="linalg.solve_sylvester", shapes=((3, 3), (1, 1), (3, 1))),
        ]
        return _with_dtypes(cases)

    def numpy_conversion_test_data(self):
        return _with_dtypes([dict(shape=(2, 2))])

    def random_distrib_complex_test_data(self):
        data = [
            dict(func_name="random.rand", args=(2,)),
            dict(func_name="random.uniform", kwargs={"size": (2,)}),
            dict(func_name="random.normal", kwargs={"size": (2,)}),
            dict(
                func_name="random.multivariate_normal",
                args=(gs.zeros(2), SPDMatrices(2).random_point()),
                kwargs={"size": (2,)},
            ),
        ]
        return data
