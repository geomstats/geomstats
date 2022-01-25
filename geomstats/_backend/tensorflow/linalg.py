"""Tensorflow based linear algebra backend."""

import scipy.optimize
import tensorflow as tf

# "Forward-import" primitives. Due to the way the 'linalg' module is exported
# in TF, this does not work with 'from tensorflow.linalg import ...'.
det = tf.linalg.det
eigh = tf.linalg.eigh
expm = tf.linalg.expm
inv = tf.linalg.inv
sqrtm = tf.linalg.sqrtm
diagonal = tf.linalg.diag_part
solve = tf.linalg.solve


def norm(x, dtype=tf.float32, **kwargs):
    x = tf.cast(x, dtype)
    return tf.linalg.norm(x, **kwargs)


def eig(*args, **kwargs):
    raise NotImplementedError


def eigvalsh(a, **_unused_kwargs):
    return tf.linalg.eigvalsh(a)


def cholesky(a, **_unused_kwargs):
    return tf.linalg.cholesky(a)


def logm(x):
    original_type = x.dtype
    x = tf.cast(x, tf.complex64)
    tf_logm = tf.linalg.logm(x)
    tf_logm = tf.cast(tf_logm, original_type)
    return tf_logm


def matrix_rank(a, **_unused_kwargs):
    return tf.linalg.matrix_rank(a)


def svd(x, full_matrices=True, compute_uv=True, **kwargs):
    is_vectorized = x.ndim == 3
    axis = (0, 2, 1) if is_vectorized else (1, 0)
    if compute_uv:
        s, u, v_t = tf.linalg.svd(x, full_matrices=full_matrices, compute_uv=compute_uv)
        return u, s, tf.transpose(v_t, perm=axis)
    return tf.linalg.svd(x, compute_uv=compute_uv)


def solve_sylvester(a, b, q):
    axes = (0, 2, 1) if a.ndim == 3 else (1, 0)
    if a.shape == b.shape:
        if tf.reduce_all(a == b) and tf.reduce_all(
            tf.abs(a - tf.transpose(a, perm=axes)) < 1e-8
        ):
            eigvals, eigvecs = eigh(a)
            if tf.reduce_all(eigvals >= 1e-8):
                tilde_q = tf.transpose(eigvecs, perm=axes) @ q @ eigvecs
                tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ tf.transpose(eigvecs, perm=axes)

            conditions = (
                a.shape[-1] >= 2
                and tf.reduce_all(eigvals[..., 0] >= -1e-8)
                and tf.reduce_all(eigvals[..., 1] >= 1e-8)
                and tf.reduce_all(tf.abs(q + tf.transpose(q, perm=axes)) < 1e-8)
            )

            if conditions:
                tilde_q = tf.transpose(eigvecs, perm=axes) @ q @ eigvecs
                safe_denominator = (
                    eigvals[..., :, None] + eigvals[..., None, :] + tf.eye(a.shape[-1])
                )
                tilde_x = tilde_q / safe_denominator
                return eigvecs @ tilde_x @ tf.transpose(eigvecs, perm=axes)

    raise RuntimeError(
        "solve_sylvester is only implemented if a = b, a symmetric and either a is "
        "positive definite or q is skew-symmetric and a is positive "
        "semi-definite with the first two eigen values positive"
    )


def quadratic_assignment(a, b, options):
    return list(scipy.optimize.quadratic_assignment(a, b, options=options).col_ind)


def qr(x, mode="reduced"):
    return tf.linalg.qr(x, full_matrices=(mode == "complete"))


def is_single_matrix_pd(mat):
    """Check if 2D square matrix is positive definite."""
    if mat.shape[0] != mat.shape[1]:
        return False
    try:
        cf = tf.linalg.cholesky(mat)
        return ~tf.math.reduce_any(tf.math.is_nan(cf)).numpy()
    except tf.errors.InvalidArgumentError as e:
        if "Cholesky decomposition was not successful" in e.message:
            return False
        raise e
