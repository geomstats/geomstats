import tensorflow as tf


def array(x):
    return tf.convert_to_tensor(x)


def ndim(x):
    x = array(x)
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def to_ndarray(x, to_ndim, axis=0):
    if ndim(x) == to_ndim - 1:
        x = tf.expand_dims(x, axis=axis)
    return x
