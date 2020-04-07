"""Tensorflow based linear algebra backend."""

import tensorflow as tf

# "Forward-import" primitives. Due to the way the 'linalg' module is exported
# in TF, this does not work with 'from tensorflow.linalg import ...'.
det = tf.linalg.det
eigh = tf.linalg.eigh
eigvalsh = tf.linalg.eigvalsh
expm = tf.linalg.expm
inv = tf.linalg.inv
norm = tf.linalg.norm
sqrtm = tf.linalg.sqrtm
set_diag = tf.linalg.set_diag
diagonal = tf.linalg.diag_part

def eig(*args, **kwargs):
    raise NotImplementedError


def logm(x):
    x = tf.cast(x, tf.complex64)
    logm = tf.linalg.logm(x)
    logm = tf.cast(logm, tf.float32)
    return logm


def svd(x):
    s, u, v_t = tf.linalg.svd(x, full_matrices=True)
    return u, s, tf.transpose(v_t, perm=(0, 2, 1))


def qr(*args, mode='reduced'):
    def qr_aux(x, mode):
        if mode == 'complete':
            aux = tf.linalg.qr(x, full_matrices=True)
        else:
            aux = tf.linalg.qr(x)

        return (aux.q, aux.r)

    qr = tf.map_fn(
        lambda x: qr_aux(x, mode),
        *args,
        dtype=(tf.float32, tf.float32))

    return qr


def powerm(x, power):
    return expm(power * logm(x))
