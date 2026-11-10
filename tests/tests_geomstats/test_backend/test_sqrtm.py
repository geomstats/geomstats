import pytest

import geomstats.backend as gs


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize(
    "dtype_name,values,roots,result_dtype",
    [
        ("complex64", [3 + 4j, 5 + 12j], [2 + 1j, 3 + 2j], "complex64"),
        ("complex128", [3 + 4j, 5 + 12j], [2 + 1j, 3 + 2j], "complex128"),
        ("float32", [-4, 9], [2j, 3], "complex64"),
        ("float64", [-4, 9], [2j, 3], "complex128"),
        ("float32", [4, 9], [2, 3], "float32"),
        ("float64", [4, 9], [2, 3], "float64"),
    ],
)
def test_sqrtm_precision(dtype_name, values, roots, result_dtype, batched):
    matrix = gs.array([[values[0], 0], [0, values[1]]], dtype=getattr(gs, dtype_name))
    expected = gs.array([[roots[0], 0], [0, roots[1]]], dtype=getattr(gs, result_dtype))
    if batched:
        matrix = gs.stack([matrix, 4 * matrix])
        expected = gs.stack([expected, 2 * expected])
    result = gs.linalg.sqrtm(matrix)
    assert result.shape == matrix.shape
    assert result.dtype == expected.dtype
    assert gs.allclose(result, expected, rtol=1e-5, atol=1e-6)
