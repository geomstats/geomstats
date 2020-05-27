import numpy as np
import tensorflow as tf


def value_and_grad(objective):
    def objective_with_grad(velocity):
        if isinstance(velocity, np.ndarray):
            velocity = tf.Variable(velocity)
        with tf.GradientTape() as t:
            t.watch(velocity)
            loss = objective(velocity)
        return loss.numpy(), t.gradient(loss, velocity).numpy()
    return objective_with_grad
