"""Tensorflow based linear algebra backend."""

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
    raise RuntimeError(
        "solve_sylvester is not implemented in tensorflow if a != b or a not"
        " Symmetric Semi Definite"
    )


def qr(x, mode='reduced'):
    return tf.linalg.qr(x, full_matrices=(mode == 'complete'))



def _is_single_matrix_pd(mat):
    """ Check if a two dimensional square matrix is 
    positive definite
    """
    try:
        ch = tf.linalg.cholesky(m)
        return True
    except tf.errors.InvalidArgumentError as e:
        if "Cholesky decomposition was not successful" in e.meesage:
            return False
        else:
            raise e

def is_pd(mat):
    """Check if matrix is positive definite matrix
    (doesn't check if its symmetric)
    """
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        return _is_single_matrix_pd(mat)
    elif mat.ndim == 2 and mat.shape[0] != mat.shape[1]:
        return False
    elif mat.ndim == 3 and mat.shape[1] == mat.shape[2]:
        return [_is_single_matrix_pd(m) for m in mat]
    elif mat.ndim == 3 and mat.shape[1] != mat.shape[2]:
        return [False] * mat.shape[0]

