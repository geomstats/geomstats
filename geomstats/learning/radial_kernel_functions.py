"""Radial kernel functions.

References
----------
https://en.wikipedia.org/wiki/Kernel_(statistics)
https://en.wikipedia.org/wiki/Radial_basis_function
"""

import geomstats.backend as gs


def uniform_radial_kernel(distance, bandwidth=1.0):
    """Uniform radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        1 / 2,
        0)
    return weight


def triangular_radial_kernel(distance, bandwidth=1.0):
    """Triangular radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        1 - scaled_distance,
        0)
    return weight


def parabolic_radial_kernel(distance, bandwidth=1.0):
    """Parabolic radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        3 / 4 * (1 - scaled_distance ** 2),
        0)
    return weight


def biweight_radial_kernel(distance, bandwidth=1.0):
    """Biweight radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        15 / 16 * (1 - scaled_distance ** 2) ** 2,
        0)
    return weight


def triweight_radial_kernel(distance, bandwidth=1.0):
    """Triweight radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        35 / 32 * (1 - scaled_distance ** 2) ** 3,
        0)
    return weight


def tricube_radial_kernel(distance, bandwidth=1.0):
    """Tricube radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        70 / 81 * (1 - scaled_distance ** 3) ** 3,
        0)
    return weight


def gaussian_radial_kernel(distance, bandwidth=1.0):
    """Gaussian radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.exp(- scaled_distance ** 2 / 2)
    weight /= (2 * gs.pi) ** (1 / 2)
    return weight


def cosine_radial_kernel(distance, bandwidth=1.0):
    """Cosine radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        gs.pi / 4 * gs.cos((gs.pi / 2) * scaled_distance),
        0)
    return weight


def logistic_radial_kernel(distance, bandwidth=1.0):
    """Logistic radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = 1 / (gs.exp(scaled_distance) + 2 + gs.exp(- scaled_distance))
    return weight


def sigmoid_radial_kernel(distance, bandwidth=1.0):
    """Sigmoid radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = 1 / (gs.exp(scaled_distance) + gs.exp(- scaled_distance))
    weight *= 2 / gs.pi
    return weight


def bump_radial_kernel(distance, bandwidth=1.0):
    """Bump radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.where(
        scaled_distance < 1,
        gs.exp(- 1 / (1 - scaled_distance ** 2)),
        0)
    return weight


def inverse_quadratic_radial_kernel(distance, bandwidth=1.0):
    """Inverse quadratic radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = 1 / (1 + scaled_distance ** 2)
    return weight


def inverse_multiquadric_radial_kernel(distance, bandwidth=1.0):
    """Inverse multiquadric radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = 1 / (1 + scaled_distance ** 2) ** (1 / 2)
    return weight


def laplacian_radial_kernel(distance, bandwidth=1.0):
    """Laplacian radial kernel.

    Parameters
    ----------
    distance : array-like
        Array of non-negative real values.
    bandwidth : float, optional (default=1.0)
        Positive scale parameter of the kernel.

    Returns
    -------
    weight : array-like
        Array of non-negative real values of the same shape than
        parameter 'distance'.
    """
    if gs.any(distance < 0):
        raise ValueError('The distance should be a non-negative real number.')
    if gs.any(bandwidth <= 0):
        raise ValueError('The bandwidth should be a positive real number.')
    scaled_distance = distance / bandwidth
    weight = gs.exp(- scaled_distance)
    return weight
