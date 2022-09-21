import logging
from functools import reduce

import geomstats.backend as gs


def equal(mat_a, mat_b, atol=gs.atol):
    """Test if matrices a and b are close.

    Parameters
    ----------
    mat_a : array-like, shape=[..., dim1, dim2]
        Matrix.
    mat_b : array-like, shape=[..., dim2, dim3]
        Matrix.
    atol : float
        Tolerance.
        Optional, default: backend atol.

    Returns
    -------
    eq : array-like, shape=[...,]
        Boolean evaluating if the matrices are close.
    """
    return gs.all(gs.isclose(mat_a, mat_b, atol=atol), (-2, -1))


def mul(*args):
    """Compute the product of matrices a1, ..., an.

    Parameters
    ----------
    a1 : array-like, shape=[..., dim_1, dim_2]
        Matrix.
    a2 : array-like, shape=[..., dim_2, dim_3]
        Matrix.
    ...
    an : array-like, shape=[..., dim_n-1, dim_n]
        Matrix.

    Returns
    -------
    mul : array-like, shape=[..., dim_1, dim_n]
        Result of the product of matrices.
    """
    return reduce(gs.matmul, args)


def bracket(mat_a, mat_b):
    """Compute the commutator of a and b, i.e. `[a, b] = ab - ba`.

    Parameters
    ----------
    mat_a : array-like, shape=[..., n, n]
        Matrix.
    mat_b : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    mat_c : array-like, shape=[..., n, n]
        Commutator.
    """
    return mul(mat_a, mat_b) - mul(mat_b, mat_a)


def transpose(mat):
    """Return the transpose of matrices.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    transpose : array-like, shape=[..., n, n]
        Transposed matrix.
    """
    is_vectorized = gs.ndim(mat) == 3
    axes = (0, 2, 1) if is_vectorized else (1, 0)
    return gs.transpose(mat, axes)


def diagonal(mat):
    """Return the diagonal of a matrix as a vector.

    Parameters
    ----------
    mat : array-like, shape=[..., m, n]
        Matrix.

    Returns
    -------
    diagonal : array-like, shape=[..., min(m, n)]
        Vector of diagonal coefficients.
    """
    return gs.diagonal(mat, axis1=-2, axis2=-1)


def is_square(mat):
    """Check if a matrix is square.

    Parameters
    ----------
    mat : array-like, shape=[..., m, n]
        Matrix.

    Returns
    -------
    is_square : array-like, shape=[...,]
        Boolean evaluating if the matrix is square.
    """
    n = mat.shape[-1]
    m = mat.shape[-2]
    return m == n


def is_diagonal(mat, atol=gs.atol):
    """Check if a matrix is square and diagonal.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.
    atol : float
        Absolute tolerance.
        Optional, default: backend atol.

    Returns
    -------
    is_diagonal : array-like, shape=[...,]
        Boolean evaluating if the matrix is square and diagonal.
    """
    is_square_ = is_square(mat)
    if not gs.all(is_square_):
        return False
    diagonal_mat = from_vector_to_diagonal_matrix(diagonal(mat))
    is_diagonal = gs.all(gs.isclose(mat, diagonal_mat, atol=atol), axis=(-2, -1))
    return is_diagonal


def is_lower_triangular(mat, atol=gs.atol):
    """Check if a matrix is lower triangular.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.
    atol : float
        Absolute tolerance.
        Optional, default : backend atol.

    Returns
    -------
    is_tril : array-like, shape=[...,]
        Boolean evaluating if the matrix is lower triangular
    """
    is_square_ = is_square(mat)
    if not is_square_:
        is_vectorized = gs.ndim(gs.array(mat)) == 3
        return gs.array([False] * len(mat)) if is_vectorized else False
    return equal(mat, gs.tril(mat), atol)


def is_upper_triangular(mat, atol=gs.atol):
    """Check if a matrix is upper triangular.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.
    atol : float
        Absolute tolerance.
        Optional, default : backend atol.

    Returns
    -------
    is_triu : array-like, shape=[...,]
        Boolean evaluating if the matrix is upper triangular.
    """
    is_square_ = is_square(mat)
    if not is_square_:
        is_vectorized = gs.ndim(gs.array(mat)) == 3
        return gs.array([False] * len(mat)) if is_vectorized else False
    return equal(mat, gs.triu(mat), atol)


def is_strictly_lower_triangular(mat, atol=gs.atol):
    """Check if a matrix is strictly lower triangular.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.
    atol : float
        Absolute tolerance.
        Optional, default : backend atol.

    Returns
    -------
    is_strictly_tril : array-like, shape=[...,]
        Boolean evaluating if the matrix is strictly lower triangular
    """
    is_square_ = is_square(mat)
    if not is_square_:
        is_vectorized = gs.ndim(mat) == 3
        return gs.array([False] * len(mat)) if is_vectorized else False
    return equal(mat, gs.tril(mat, k=-1), atol)


def is_strictly_upper_triangular(mat, atol=gs.atol):
    """Check if a matrix is strictly upper triangular.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.
    atol : float
        Absolute tolerance.
        Optional, default : backend atol.

    Returns
    -------
    is_strictly_triu : array-like, shape=[...,]
        Boolean evaluating if the matrix is strictly upper triangular
    """
    is_square_ = is_square(mat)
    if not is_square_:
        is_vectorized = gs.ndim(gs.array(mat)) == 3
        return gs.array([False] * len(mat)) if is_vectorized else False
    return equal(mat, gs.triu(mat, k=1))


def is_symmetric(mat, atol=gs.atol):
    """Check if a matrix is symmetric.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.
    atol : float
        Absolute tolerance.
        Optional, default: backend atol.

    Returns
    -------
    is_sym : array-like, shape=[...,]
        Boolean evaluating if the matrix is symmetric.
    """
    is_square_ = is_square(mat)
    if not is_square_:
        is_vectorized = gs.ndim(gs.array(mat)) == 3
        return gs.array([False] * len(mat)) if is_vectorized else False
    return equal(mat, transpose(mat), atol)


def is_pd(mat):
    """Check if a matrix is positive definite.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.
    atol : float
        Absolute tolerance.
        Optional, default: backend atol.

    Returns
    -------
    is_pd : array-like, shape=[...,]
        Boolean evaluating if the matrix is positive definite.
    """
    if mat.ndim == 2:
        return gs.array(gs.linalg.is_single_matrix_pd(mat))
    return gs.array([gs.linalg.is_single_matrix_pd(m) for m in mat])


def is_spd(mat, atol=gs.atol):
    """Check if a matrix is symmetric positive definite.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.
    atol : float
        Absolute tolerance.
        Optional, default: backend atol.

    Returns
    -------
    is_spd : array-like, shape=[...,]
        Boolean evaluating if the matrix is symmetric positive definite.
    """
    is_spd = gs.logical_and(is_symmetric(mat, atol), is_pd(mat))
    return is_spd


def is_skew_symmetric(mat, atol=gs.atol):
    """Check if a matrix is skew symmetric.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.
    atol : float
        Absolute tolerance.
        Optional, default: backend atol.

    Returns
    -------
    is_skew_sym : array-like, shape=[...,]
        Boolean evaluating if the matrix is skew-symmetric.
    """
    is_square_ = is_square(mat)
    if not is_square_:
        is_vectorized = gs.ndim(gs.array(mat)) == 3
        return gs.array([False] * len(mat)) if is_vectorized else False
    return equal(mat, -transpose(mat), atol)


def to_diagonal(mat):
    """Make a matrix diagonal.

    Make a matrix diagonal by zeroing out non
    diagonal elements.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    diag : array-like, shape=[..., n, n]
    """
    return to_upper_triangular(to_lower_triangular(mat))


def to_lower_triangular(mat):
    """Make a matrix lower triangular.

    Make a matrix lower triangular by zeroing
    out upper elements.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    tril : array-like, shape=[..., n, n]
        Lower  triangular matrix.
    """
    return gs.tril(mat)


def to_upper_triangular(mat):
    """Make a matrix upper triangular.

    Make a matrix upper triangular by zeroing
    out lower elements.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    triu : array-like, shape=[..., n, n]
    """
    return gs.triu(mat)


def to_strictly_lower_triangular(mat):
    """Make a matrix strictly lower triangular.

    Make a matrix stricly lower triangular by zeroing
    out upper and diagonal elements.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    tril : array-like, shape=[..., n, n]
        Lower  triangular matrix.
    """
    return gs.tril(mat, k=-1)


def to_strictly_upper_triangular(mat):
    """Make a matrix stritcly upper triangular.

    Make a matrix strictly upper triangular by zeroing
    out lower and diagonal elements.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    triu : array-like, shape=[..., n, n]
    """
    return gs.triu(mat, k=1)


def to_symmetric(mat):
    """Make a matrix symmetric.

    Make a matrix suymmetric by averaging it
    with its transpose.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    sym : array-like, shape=[..., n, n]
        Symmetric matrix.
    """
    return 1 / 2 * (mat + transpose(mat))


def to_skew_symmetric(mat):
    """Make a matrix skew-symmetric.

    Make matrix skew-symmetric by averaging it
    with minus its transpose.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    skew_sym : array-like, shape=[..., n, n]
        Skew-symmetric matrix.
    """
    return 1 / 2 * (mat - transpose(mat))


def to_lower_triangular_diagonal_scaled(mat, K=2.0):
    """Make a matrix lower triangular.

    Make matrix lower triangular by zeroing out
    upper elements and divide diagonal by factor K.

    Parameters
    ----------
    mat : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    tril : array-like, shape=[..., n, n]
        Lower  triangular matrix.
    """
    slt = to_strictly_lower_triangular(mat)
    diag = to_diagonal(mat) / K
    return slt + diag


def congruent(mat_1, mat_2):
    r"""Compute the congruent action of mat_2 on mat_1.

    This is :math:`mat\_2 \ mat\_1 \ mat\_2^T`.

    Parameters
    ----------
    mat_1 : array-like, shape=[..., n, n]
        Matrix.
    mat_2 : array-like, shape=[..., n, n]
        Matrix.

    Returns
    -------
    cong : array-like, shape=[..., n, n]
        Result of the congruent action.
    """
    return mul(mat_2, mat_1, transpose(mat_2))


def frobenius_product(mat_1, mat_2):
    """Compute Frobenius inner-product of two matrices.

    The `einsum` function is used to avoid computing a matrix product. It
    is also faster than using a sum an element-wise product.

    Parameters
    ----------
    mat_1 : array-like, shape=[..., m, n]
        Matrix.
    mat_2 : array-like, shape=[..., m, n]
        Matrix.

    Returns
    -------
    product : array-like, shape=[...,]
        Frobenius inner-product of mat_1 and mat_2
    """
    return gs.einsum("...ij,...ij->...", mat_1, mat_2)


def trace_product(mat_1, mat_2):
    """Compute trace of the product of two matrices.

    The `einsum` function is used to avoid computing a matrix product. It
    is also faster than using a sum an element-wise product.

    Parameters
    ----------
    mat_1 : array-like, shape=[..., m, n]
        Matrix.
    mat_2 : array-like, shape=[..., m, n]
        Matrix.

    Returns
    -------
    product : array-like, shape=[...,]
        Frobenius inner-product of mat_1 and mat_2
    """
    return gs.einsum("...ij,...ji->...", mat_1, mat_2)


def from_vector_to_diagonal_matrix(vector, num_diag=0):
    """Create diagonal matrices from rows of a matrix.

    Parameters
    ----------
    vector : array-like, shape=[m, n]
    num_diag : int
        number of diagonal in result matrix. If 0, the result matrix is a
        diagonal matrix; if positive, the result matrix has an upper-right
        non-zero diagonal; if negative, the result matrix has a lower-left
        non-zero diagonal.
        Optional, Default: 0.

    Returns
    -------
    diagonals : array-like, shape=[m, n, n]
        3-dimensional array where the `i`-th n-by-n array `diagonals[i, :, :]`
        is a diagonal matrix containing the `i`-th row of `vector`.
    """
    num_columns = gs.shape(vector)[-1]
    identity = gs.eye(num_columns)
    identity = gs.cast(identity, vector.dtype)
    diagonals = gs.einsum("...i,ij->...ij", vector, identity)
    diagonals = gs.to_ndarray(diagonals, to_ndim=3)
    num_lines = diagonals.shape[0]
    if num_diag > 0:
        left_zeros = gs.zeros((num_lines, num_columns, num_diag))
        lower_zeros = gs.zeros((num_lines, num_diag, num_columns + num_diag))
        diagonals = gs.concatenate((left_zeros, diagonals), axis=2)
        diagonals = gs.concatenate((diagonals, lower_zeros), axis=1)
    elif num_diag < 0:
        num_diag = gs.abs(num_diag)
        right_zeros = gs.zeros((num_lines, num_columns, num_diag))
        upper_zeros = gs.zeros((num_lines, num_diag, num_columns + num_diag))
        diagonals = gs.concatenate((diagonals, right_zeros), axis=2)
        diagonals = gs.concatenate((upper_zeros, diagonals), axis=1)
    return gs.squeeze(diagonals) if gs.ndim(vector) == 1 else diagonals


def flip_determinant(matrix, det):
    """Change sign of the determinant if it is negative.

    For a batch of matrices, multiply the matrices which have negative
    determinant by a diagonal matrix :math:`diag(1,...,1,-1) from the right.
    This changes the sign of the last column of the matrix.

    Parameters
    ----------
    matrix : array-like, shape=[...,n ,m]
        Matrix to transform.

    det : array-like, shape=[...]
        Determinant of matrix, or any other scalar to use as threshold to
        determine whether to change the sign of the last column of matrix.

    Returns
    -------
    matrix_flipped : array-like, shape=[..., n, m]
        Matrix with the sign of last column changed if det < 0.
    """
    if gs.any(det < 0):
        ones = gs.ones(matrix.shape[-1])
        reflection_vec = gs.concatenate([ones[:-1], gs.array([-1.0])], axis=0)
        mask = gs.cast(det < 0, matrix.dtype)
        sign = mask[..., None] * reflection_vec + (1.0 - mask)[..., None] * ones
        return gs.einsum("...ij,...j->...ij", matrix, sign)
    return matrix


def align_matrices(point, base_point):
    """Align matrices.

    Find the optimal rotation R in SO(m) such that the base point and
    R.point are well positioned.

    Parameters
    ----------
    point : array-like, shape=[..., m, n]
        Point on the manifold.
    base_point : array-like, shape=[..., m, n]
        Point on the manifold.

    Returns
    -------
    aligned : array-like, shape=[..., m, n]
        R.point.
    """
    mat = gs.matmul(transpose(point), base_point)
    left, singular_values, right = gs.linalg.svd(mat, full_matrices=False)
    det = gs.linalg.det(mat)
    conditioning = (
        singular_values[..., -2] + gs.sign(det) * singular_values[..., -1]
    ) / singular_values[..., 0]
    if gs.any(conditioning < gs.atol):
        logging.warning(
            f"Singularity close, ill-conditioned matrix "
            f"encountered: "
            f"{conditioning[conditioning < 1e-10]}"
        )
    if gs.any(gs.isclose(conditioning, 0.0)):
        logging.warning("Alignment matrix is not unique.")
    flipped = flip_determinant(transpose(right), det)
    return mul(point, left, transpose(flipped))


def norm(vector):
    """Compute norm of a matrix.

    Norm of a matrix associated to the Frobenius inner product.

    Parameters
    ----------
    vector : array-like, shape=[..., dim]
        Vector.

    Returns
    -------
    norm : array-like, shape=[...,]
        Norm.
    """
    return gs.linalg.norm(vector, axis=(-2, -1))
