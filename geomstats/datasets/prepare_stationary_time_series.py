"""The Burg algorithms.

Lead author: Yann Cabanes.
"""

import geomstats.backend as gs


def beta(k, i, gamma):
    return gamma * (2 * gs.pi) ** 2 * (k - i) ** 2


def burg_regularized_for_one_time_series(data, model_order=4, gamma="default"):

    if gs.all(data == 0):
        return data
    n_pulses = data.shape[0]
    model_order = int(min(model_order, n_pulses - 1))
    if gamma == "default":
        gamma = 5e-8 * gs.linalg.norm(data) ** 2 / n_pulses

    """Initialization"""
    power_and_reflection_coefficients = gs.zeros([model_order + 1], dtype=complex)
    power_and_reflection_coefficients[0] = gs.real(1 / n_pulses * data.conj().T @ data)
    f = gs.zeros([model_order + 1, n_pulses], dtype=complex)
    b = gs.zeros([model_order + 1, n_pulses], dtype=complex)
    a = gs.zeros([model_order + 1, model_order + 1], dtype=complex)
    f[0, :] = data
    b[0, :] = data
    a[0, :] = gs.ones([model_order + 1])
    """Iteration"""
    for i in range(1, model_order + 1):
        sum1 = 0
        for k in range(i, n_pulses):
            sum1 += f[i - 1, k] * gs.conj(b[i - 1, k - 1])
        sum2 = 0
        for k in range(1, i):
            sum2 += beta(k, i, gamma) * a[k, i - 1] * a[i - k, i - 1]
        sum3 = 0
        for k in range(i, n_pulses):
            sum3 += abs(f[i - 1, k]) ** 2 + abs(b[i - 1, k - 1]) ** 2
        sum4 = 0
        for k in range(i):
            sum4 += beta(k, i, gamma) * abs(a[k, i - 1]) ** 2
        power_and_reflection_coefficients[i] = -(
            2 / (n_pulses - i) * sum1 + 2 * sum2
        ) / (1 / (n_pulses - i) * sum3 + 2 * sum4)
        a[i, i] = power_and_reflection_coefficients[i]
        if i <= model_order - 1:
            for k in range(1, i):
                a[k, i] = a[k, i - 1] + power_and_reflection_coefficients[i] * gs.conj(
                    a[i - k, i - 1]
                )
            for k in range(i, n_pulses):
                f[i, k] = (
                    f[i - 1, k] + power_and_reflection_coefficients[i] * b[i - 1, k - 1]
                )
                b[i, k] = (
                    b[i - 1, k - 1]
                    + gs.conj(power_and_reflection_coefficients[i]) * f[i - 1, k]
                )
    return power_and_reflection_coefficients


def burg_regularized(data, model_order=4, gamma="default"):
    n_samples = data.shape[0]
    n_pulses = data.shape[1]
    model_order = int(min(model_order, n_pulses - 1))
    power_and_reflection_coefficients = gs.zeros(
        [n_samples, model_order + 1], dtype=complex
    )
    for i_sample in range(n_samples):
        power_and_reflection_coefficients[
            i_sample, :
        ] = burg_regularized_for_one_time_series(data[i_sample, :], model_order, gamma)
    return power_and_reflection_coefficients
