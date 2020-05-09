"""Tensorflow based linear algebra backend."""

import tensorflow as tf

# "Forward-import" primitives. Due to the way the 'linalg' module is exported
# in TF, this does not work with 'from tensorflow.linalg import ...'.
det = tf.linalg.det
eigh = tf.linalg.eigh
eigvalsh = tf.linalg.eigvalsh
expm = tf.linalg.expm
inv = tf.linalg.inv
sqrtm = tf.linalg.sqrtm
diagonal = tf.linalg.diag_part


def norm(x, dtype=tf.float32, **kwargs):
    x = tf.cast(x, dtype)
    return tf.linalg.norm(x, **kwargs)


def eig(*args, **kwargs):
    raise NotImplementedError


def logm(x):
    original_type = x.dtype
    x = tf.cast(x, tf.complex64)
    tf_logm = tf.linalg.logm(x)
    tf_logm = tf.cast(tf_logm, original_type)
    return tf_logm


def svd(x):
    s, u, v_t = tf.linalg.svd(x, full_matrices=True)
    return u, s, tf.transpose(v_t, perm=(0, 2, 1))


def qr(*args, mode='reduced'):
    def qr_aux(x, mode):
        if mode == 'complete':
            aux = tf.linalg.qr(x, full_matrices=True)
        else:
            aux = tf.linalg.qr(x)

        return aux.q, aux.r

    result = tf.map_fn(
        lambda x: qr_aux(x, mode),
        *args,
        dtype=(tf.float32, tf.float32))

    return result


def powerm(x, power):
    return expm(power * logm(x))
