from ._dispatch import _common
from ._dispatch import numpy as _np
from ._dispatch import scipy as _scipy

_to_ndarray = _common.to_ndarray
_cast_fout_to_input_dtype = _common._cast_fout_to_input_dtype
_cast_out_to_input_dtype = _common._cast_out_to_input_dtype
atol = _common.atol


def _is_symmetric(x, tol=atol):
    new_x = _to_ndarray(x, to_ndim=3)
    return (_np.abs(new_x - _np.transpose(new_x, axes=(0, 2, 1))) < tol).all()


def _is_hermitian(x, tol=atol):
    new_x = _to_ndarray(x, to_ndim=3)
    return (_np.abs(new_x - _np.conj(_np.transpose(new_x, axes=(0, 2, 1)))) < tol).all()


_diag_vec = _np.vectorize(_np.diag, signature="(n)->(n,n)")

_logm_vec = _cast_fout_to_input_dtype(
    target=_np.vectorize(_scipy.linalg.logm, signature="(n,m)->(n,m)")
)


def logm(x):
    ndim = x.ndim
    new_x = _to_ndarray(x, to_ndim=3)

    if _is_symmetric(new_x) and new_x.dtype not in [_np.complex64, _np.complex128]:
        eigvals, eigvecs = _np.linalg.eigh(new_x)
        if (eigvals > 0).all():
            eigvals = _np.log(eigvals)
            eigvals = _diag_vec(eigvals)
            transp_eigvecs = _np.transpose(eigvecs, axes=(0, 2, 1))
            result = _np.matmul(eigvecs, eigvals)
            result = _np.matmul(result, transp_eigvecs)
        else:
            result = _logm_vec(new_x)
    else:
        result = _logm_vec(new_x)

    if ndim == 2:
        return result[0]
    return result


def solve_sylvester(a, b, q, tol=atol):
    if a.shape == b.shape:
        axes = (0, 2, 1) if a.ndim == 3 else (1, 0)
        if _np.all(_np.isclose(a, b)) and _np.all(
            _np.abs(a - _np.transpose(a, axes)) < tol
        ):
            eigvals, eigvecs = _np.linalg.eigh(a)
            if _np.all(eigvals >= tol):
                tilde_q = _np.transpose(eigvecs, axes) @ q @ eigvecs
                tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ _np.transpose(eigvecs, axes)

    return _np.vectorize(
        _scipy.linalg.solve_sylvester, signature="(m,m),(n,n),(m,n)->(m,n)"
    )(a, b, q)


@_cast_fout_to_input_dtype
def sqrtm(x):
    return _np.vectorize(_scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x)


def quadratic_assignment(a, b, options):
    return list(_scipy.optimize.quadratic_assignment(a, b, options=options).col_ind)


def qr(*args, **kwargs):
    return _np.vectorize(
        _np.linalg.qr, signature="(n,m)->(n,k),(k,m)", excluded=["mode"]
    )(*args, **kwargs)


def is_single_matrix_pd(mat):
    """Check if 2D square matrix is positive definite."""
    if mat.shape[0] != mat.shape[1]:
        return False
    if mat.dtype in [_np.complex64, _np.complex128]:
        if not _is_hermitian(mat):
            return False
        eigvals = _np.linalg.eigvalsh(mat)
        return _np.min(_np.real(eigvals)) > 0
    try:
        _np.linalg.cholesky(mat)
        return True
    except _np.linalg.LinAlgError as e:
        if e.args[0] == "Matrix is not positive definite":
            return False
        raise e


@_cast_out_to_input_dtype
def fractional_matrix_power(A, t):
    if A.ndim == 2:
        return _scipy.linalg.fractional_matrix_power(A, t)

    return _np.stack([_scipy.linalg.fractional_matrix_power(A_, t) for A_ in A])
